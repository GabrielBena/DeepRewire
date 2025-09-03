"""
The optimizers for the rewiring network
"""

import torch
from torch.optim.optimizer import Optimizer, required
import numpy as np


class DEEPR(Optimizer):
    """
    Deep-rewiring oftimizer with hard constraint on number of connections
    """

    def __init__(
        self,
        params,
        nc=required,
        lr=0.05,
        l1=1e-4,
        reset_val=0.0,
        max_val=2.0,
        grad_clip=None,
        temp=None,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if nc is not required and nc < 0.0 or not isinstance(nc, int):
            raise ValueError(f"Invalid number of connections: {nc}")
        if l1 < 0.0:
            raise ValueError(f"Invalid L1 regularization term: {l1}")
        if reset_val < 0.0:
            raise ValueError(f"Invalid reset value: {reset_val}")
        if max_val < reset_val:
            raise ValueError(f"Invalid max value: {max_val}")

        if temp is None:
            temp = lr / 210

        self.nc = nc

        defaults = dict(
            lr=lr,
            l1=l1,
            temp=temp,
            reset_val=reset_val,
            max_val=max_val,
            grad_clip=grad_clip,
        )
        super().__init__(params, defaults)

        # count parameters
        self.n_parameters = 0
        for group in self.param_groups:
            for p in group["params"]:
                if not p.requires_grad:
                    continue
                self.n_parameters += p.numel()

        if self.nc > self.n_parameters:
            raise ValueError(
                "Number of connections can't be bigger than number"
                + f"of parameters: nc:{nc} np:{self.n_parameters}"
            )

        activate_indices = self.sample_unique_indices(self.nc, self.n_parameters)
        self.init_activation(activate_indices)

    def sample_unique_indices(self, length, max_int):
        if length > max_int:
            raise ValueError(
                "Cannot sample more unique indices than the size of the range."
            )

        selected_indices = set()

        while len(selected_indices) < length:
            new_indices = torch.randint(0, max_int, (length - len(selected_indices),))
            selected_indices.update(new_indices.tolist())

        return torch.tensor(list(selected_indices))

    def init_activation(self, activate_indices):
        """
        Function to initialize activation by flipping the sign of the selected indices.
        """
        remaining_indices = activate_indices

        for group in self.param_groups:
            for p in group["params"]:
                if not p.requires_grad:
                    continue

                num_elements = p.data.numel()

                p.data = -torch.abs(p.data)

                p_data_flat = p.data.view(-1)

                in_current_param_mask = remaining_indices < num_elements
                current_indices = remaining_indices[in_current_param_mask]

                if current_indices.numel() > 0:
                    p_data_flat[current_indices] *= -1
                    p_data_flat[current_indices] = torch.clamp(
                        p_data_flat[current_indices],
                        min=group["reset_val"],
                        max=group["max_val"],
                    )

                    remaining_indices = remaining_indices[~in_current_param_mask]

                remaining_indices -= num_elements

                if remaining_indices.numel() == 0:
                    break

    def attempt_activation(self, candidate_indices):
        """
        Function will activate connections if previously inactive and return the number of activations.
        """
        activations = 0
        remaining_indices = candidate_indices

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                p_data_flat = p.data.view(-1)
                num_elements = p_data_flat.numel()

                in_current_param_mask = remaining_indices < num_elements
                current_indices = remaining_indices[in_current_param_mask]

                if current_indices.numel() > 0:
                    selected_values = p_data_flat[current_indices]

                    to_activate_mask = selected_values < 0
                    to_activate_indices = current_indices[to_activate_mask]

                    if to_activate_indices.numel() > 0:
                        p_data_flat[to_activate_indices] = group["reset_val"]
                        activations += to_activate_indices.numel()

                remaining_indices = (
                    remaining_indices[~in_current_param_mask] - num_elements
                )

                if remaining_indices.numel() == 0:
                    break

        return activations

    def step(self, closure=None):
        """Performs a single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        active_connections = 0
        idx_counter = 0
        for group in self.param_groups:
            lr = group["lr"]
            l1 = group["l1"]
            temp = group["temp"]
            sqrt_temp = (2 * lr * temp) ** 0.5

            for p in group["params"]:
                if p.grad is None:
                    continue
                if np.isnan(p.grad.data.detach().cpu().numpy()).any():
                    raise ValueError("NaN in gradient")
                if np.isnan(p.data.detach().cpu().numpy()).any():
                    raise ValueError("NaN in weights")

                grad = p.grad.data
                grad = torch.nan_to_num(grad, nan=0.0, posinf=0.0, neginf=0.0)

                if group["grad_clip"] is not None:
                    grad = grad.clamp(min=-group["grad_clip"], max=group["grad_clip"])
                noise = sqrt_temp * torch.randn_like(p.data)
                mask = p.data >= 0
                p.data += mask.float() * (-lr * (grad + l1) + noise)

                active_connections += torch.sum(p.data > 0).item()
                idx_counter += p.data.numel()

        # look how many connections are inactive and activate if necessary.
        diff = self.nc - active_connections
        while diff > 0:
            candidate_indices = torch.randint(
                low=0, high=self.n_parameters, size=(diff,)
            )
            candidate_indices = candidate_indices.to(
                self.param_groups[0]["params"][0].device
            )
            diff -= self.attempt_activation(candidate_indices)

        return loss


class SoftDEEPR(Optimizer):
    """
    Deep-rewiring oftimizer with soft constraint on number of connections
    """

    def __init__(self, params, lr=0.05, l1=1e-5, temp=None, min_weight=None):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if l1 < 0.0:
            raise ValueError(f"Invalid L1 regularization term: {l1}")
        if temp is None:
            temp = lr * l1**2 / 18
        if min_weight is None:
            min_weight = -3 * l1

        defaults = dict(lr=lr, l1=l1, temp=temp, min_weight=min_weight)
        super().__init__(params, defaults)

    def step(self, closure=None):
        """Performs a single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            l1 = group["l1"]
            temp = group["temp"]
            min_weight = group["min_weight"]
            sqrt_temp = (2 * lr * temp) ** 0.5

            for p in group["params"]:
                if p.grad is None:
                    if p.requires_grad:
                        print("grad is none")
                    continue

                grad = p.grad.data
                noise = sqrt_temp * torch.randn_like(p.data)
                mask = p.data >= 0

                """
                p.data += mask.float() * (-lr * (grad + l1) + noise)
                p.data += (~mask).float() * noise.clamp(min=min_weight)
                """

                # this is how its done in the paper i think:
                p.data += noise - mask.float() * lr * (grad + l1)
                p.data = p.data.clamp(min=min_weight)
        return loss


class SoftDEEPRWrapper(Optimizer):
    """
    Deep-rewiring oftimizer with soft constraint on number of connections
    """

    def __init__(
        self,
        params,
        base_optim_class,
        l1=1e-5,
        temp=None,
        min_weight=None,
        **optim_kwargs,
    ):
        params = list(params)
        self.base_optim = base_optim_class(params, **optim_kwargs)
        lr = optim_kwargs.get("lr", 0.005)
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if l1 < 0.0:
            raise ValueError(f"Invalid L1 regularization term: {l1}")
        if temp is None:
            temp = lr * l1**2 / 18
        if min_weight is None:
            min_weight = -3 * l1

        defaults = dict(lr=lr, l1=l1, temp=temp, min_weight=min_weight)
        super(SoftDEEPRWrapper, self).__init__(params, defaults)

    def step(self, closure=None):
        """Performs a single optimization step."""
        for group in self.param_groups:
            lr = group["lr"]
            l1 = group["l1"]
            temp = group["temp"]
            sqrt_temp = (2 * lr * temp) ** 0.5

            for p in group["params"]:
                if p.grad is None:
                    continue

                noise = sqrt_temp * torch.randn_like(p.data)
                mask = p.data >= 0
                p.grad.data *= mask.float()
                p.data += noise - mask.float() * lr * l1

        loss = self.base_optim.step(closure=closure)

        for group in self.param_groups:
            min_weight = group["min_weight"]
            for p in group["params"]:
                if p.grad is None:
                    continue

                p.data = p.data.clamp(min=min_weight)

        return loss


class DEEPR_MultiGroup(Optimizer):
    def __init__(self, params, nc=required, lr=0.05, l1=1e-4, reset_val=0.0, temp=None):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if l1 < 0.0:
            raise ValueError(f"Invalid L1 regularization term: {l1}")
        if reset_val < 0.0:
            raise ValueError(f"Invalid reset value: {reset_val}")

        if temp is None:
            temp = lr / 210

        # Modified to handle nc as either int or dict
        self.nc = nc
        defaults = dict(lr=lr, l1=l1, temp=temp, reset_val=reset_val)
        super(DEEPR_MultiGroup, self).__init__(params, defaults)

        # Count parameters per group and validate constraints
        self.group_parameters = []
        for group in self.param_groups:
            group_params = 0
            for p in group["params"]:
                if p.requires_grad:
                    group_params += p.numel()
            self.group_parameters.append(group_params)

            # Handle nc as either global int or dict with group indices
            if isinstance(nc, dict):
                group["nc"] = nc.get(len(self.group_parameters) - 1, required)
                if group["nc"] is required:
                    raise ValueError(
                        f"Missing nc for parameter group {len(self.group_parameters) - 1}"
                    )
                if group["nc"] > group_params:
                    raise ValueError(
                        f"Group {len(self.group_parameters) - 1}: nc ({group['nc']}) > parameters ({group_params})"
                    )
            else:
                group["nc"] = nc
                if group["nc"] > group_params:
                    raise ValueError(
                        f"Group {len(self.group_parameters) - 1}: nc ({group['nc']}) > parameters ({group_params})"
                    )

        # Initialize each group separately
        for group_idx, group in enumerate(self.param_groups):
            activate_indices = self.sample_unique_indices(
                group["nc"], self.group_parameters[group_idx]
            )
            self.init_activation_group(activate_indices, group)

    def init_activation_group(self, activate_indices, group):
        """Initialize activation for a single parameter group"""
        remaining_indices = activate_indices

        for p in group["params"]:
            if not p.requires_grad:
                continue

            num_elements = p.data.numel()
            p.data = -torch.abs(p.data)
            p_data_flat = p.data.view(-1)

            in_current_param_mask = remaining_indices < num_elements
            current_indices = remaining_indices[in_current_param_mask]

            if current_indices.numel() > 0:
                p_data_flat[current_indices] *= -1
                p_data_flat[current_indices] = torch.clamp(
                    p_data_flat[current_indices], min=group["reset_val"]
                )
                remaining_indices = (
                    remaining_indices[~in_current_param_mask] - num_elements
                )

    def step(self, closure=None):
        """Performs a single optimization step with per-group constraints"""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # Handle each group separately
        for group_idx, group in enumerate(self.param_groups):
            lr = group["lr"]
            l1 = group["l1"]
            temp = group["temp"]
            sqrt_temp = (2 * lr * temp) ** 0.5

            active_connections = 0
            param_offset = 0

            # Update weights and count active connections
            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad.data
                noise = sqrt_temp * torch.randn_like(p.data)
                mask = p.data >= 0
                p.data += mask.float() * (-lr * (grad + l1) + noise)

                active_connections += torch.sum(p.data > 0).item()
                param_offset += p.data.numel()

            # Activate additional connections if needed for this group
            diff = group["nc"] - active_connections
            while diff > 0:
                candidate_indices = torch.randint(
                    low=0, high=self.group_parameters[group_idx], size=(diff,)
                )
                candidate_indices = candidate_indices.to(p.device)
                diff -= self.attempt_activation_group(candidate_indices, group)

        return loss

    def attempt_activation_group(self, candidate_indices, group):
        """Attempt activation for a single parameter group"""
        activations = 0
        remaining_indices = candidate_indices

        for p in group["params"]:
            if p.grad is None:
                continue

            p_data_flat = p.data.view(-1)
            num_elements = p_data_flat.numel()

            in_current_param_mask = remaining_indices < num_elements
            current_indices = remaining_indices[in_current_param_mask]

            if current_indices.numel() > 0:
                selected_values = p_data_flat[current_indices]
                to_activate_mask = selected_values < 0
                to_activate_indices = current_indices[to_activate_mask]

                if to_activate_indices.numel() > 0:
                    p_data_flat[to_activate_indices] = group["reset_val"]
                    activations += to_activate_indices.numel()

            remaining_indices = remaining_indices[~in_current_param_mask] - num_elements

        return activations


class SoftDEEPR_MultiGroup(Optimizer):
    """
    Deep-rewiring optimizer with soft constraint on number of connections,
    supporting different L1 penalties per parameter group
    """

    def __init__(self, params, lr=0.05, l1=1e-5, temp=None, min_weight=None):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")

        # Handle l1 as either float or dict
        self.l1 = l1
        defaults = dict(lr=lr)
        super().__init__(params, defaults)

        # Set up group-specific parameters
        for group_idx, group in enumerate(self.param_groups):
            # Get group-specific L1 penalty
            if isinstance(l1, dict):
                group_l1 = l1.get(group_idx, 1e-5)
            else:
                group_l1 = l1

            if group_l1 < 0.0:
                raise ValueError(
                    f"Invalid L1 regularization term for group {group_idx}: {group_l1}"
                )

            # Set group-specific parameters
            group["l1"] = group_l1
            group["temp"] = temp if temp is not None else lr * group_l1**2 / 18
            group["min_weight"] = (
                min_weight if min_weight is not None else -3 * group_l1
            )

    def step(self, closure=None):
        """Performs a single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            l1 = group["l1"]
            temp = group["temp"]
            min_weight = group["min_weight"]
            sqrt_temp = (2 * lr * temp) ** 0.5

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                noise = sqrt_temp * torch.randn_like(p.data)

                mask = p.data >= 0

                # Update weights with group-specific parameters
                p.data += noise - mask.float() * lr * (grad + l1)
                p.data = p.data.clamp(min=min_weight)

        return loss


if __name__ == "__main__":
    pass
