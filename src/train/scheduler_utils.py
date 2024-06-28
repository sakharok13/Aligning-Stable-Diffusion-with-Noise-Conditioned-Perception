import torch
from diffusers.schedulers.scheduling_ddpm import DDPMSchedulerOutput


def _get_variance(self, t, prev_t, predicted_variance=None, variance_type=None):
    alpha_prod_t = self.alphas_cumprod[t]
    alpha_prod_t_prev = self.alphas_cumprod[prev_t] # if prev_t >= 0 else self.one
    current_beta_t = 1 - alpha_prod_t / alpha_prod_t_prev

    # For t > 0, compute predicted variance βt (see formula (6) and (7) from https://arxiv.org/pdf/2006.11239.pdf)
    # and sample from it to get previous sample
    # x_{t-1} ~ N(pred_prev_sample, variance) == add variance to pred_sample
    variance = (1 - alpha_prod_t_prev) / (1 - alpha_prod_t) * current_beta_t

    # we always take the log of variance, so clamp it to ensure it's not 0
    variance = torch.clamp(variance, min=1e-20)

    if variance_type is None:
        variance_type = self.config.variance_type

    # hacks - were probably added for training stability
    if variance_type == "fixed_small":
        variance = variance
    # for rl-diffuser https://arxiv.org/abs/2205.09991
    elif variance_type == "fixed_small_log":
        variance = torch.log(variance)
        variance = torch.exp(0.5 * variance)
    elif variance_type == "fixed_large":
        variance = current_beta_t
    elif variance_type == "fixed_large_log":
        # Glide max_log
        variance = torch.log(current_beta_t)
    elif variance_type == "learned":
        return predicted_variance
    elif variance_type == "learned_range":
        min_log = torch.log(variance)
        max_log = torch.log(current_beta_t)
        frac = (predicted_variance + 1) / 2
        variance = frac * max_log + (1 - frac) * min_log

    return variance


def unsqueeze_like(tensor: torch.Tensor, like: torch.Tensor):
    # https://pytorch-forecasting.readthedocs.io/en/latest/api/pytorch_forecasting.utils.unsqueeze_like.html
    n_unsqueezes = like.ndim - tensor.ndim
    if n_unsqueezes < 0:
        raise ValueError(f"tensor.ndim={tensor.ndim} > like.ndim={like.ndim}")
    elif n_unsqueezes == 0:
        return tensor
    else:
        return tensor[(...,) + (None,) * n_unsqueezes]


def ddpm_scheduler_step_batched(
    self,
    model_output,
    timestep,
    previous_timestep,
    sample,
    variance_noise,
    return_dict=True,
):
    t = timestep
    prev_t = previous_timestep

    if model_output.shape[1] == sample.shape[1] * 2 and self.variance_type in ["learned", "learned_range"]:
        model_output, predicted_variance = torch.split(model_output, sample.shape[1], dim=1)
    else:
        predicted_variance = None

    # 1. compute alphas, betas
    alpha_prod_t = self.alphas_cumprod[t]
    alpha_prod_t_prev = self.alphas_cumprod[prev_t] # if prev_t >= 0 else self.one
    beta_prod_t = 1 - alpha_prod_t
    beta_prod_t_prev = 1 - alpha_prod_t_prev
    current_alpha_t = alpha_prod_t / alpha_prod_t_prev
    current_beta_t = 1 - current_alpha_t

    alpha_prod_t = unsqueeze_like(alpha_prod_t, sample).to(sample.dtype)
    alpha_prod_t_prev = unsqueeze_like(alpha_prod_t_prev, sample).to(sample.dtype)
    beta_prod_t = unsqueeze_like(beta_prod_t, sample).to(sample.dtype)
    beta_prod_t_prev = unsqueeze_like(beta_prod_t_prev, sample).to(sample.dtype)
    current_alpha_t = unsqueeze_like(current_alpha_t, sample).to(sample.dtype)
    current_beta_t = unsqueeze_like(current_beta_t, sample).to(sample.dtype)

    # 2. compute predicted original sample from predicted noise also called
    # "predicted x_0" of formula (15) from https://arxiv.org/pdf/2006.11239.pdf
    pred_original_noise = None
    if self.config.prediction_type == "epsilon":
        pred_original_sample = (sample - beta_prod_t ** (0.5) * model_output) / alpha_prod_t ** (0.5)
    elif self.config.prediction_type == "sample":
        pred_original_sample = model_output
    elif self.config.prediction_type == "v_prediction":
        pred_original_sample = (alpha_prod_t**0.5) * sample - (beta_prod_t**0.5) * model_output
        pred_original_noise = (alpha_prod_t**0.5) * model_output + (beta_prod_t**0.5) * sample
    else:
        raise ValueError(
            f"prediction_type given as {self.config.prediction_type} must be one of `epsilon`, `sample` or"
            " `v_prediction`  for the DDPMScheduler."
        )

    # 3. Clip or threshold "predicted x_0"
    if self.config.thresholding:
        pred_original_sample = self._threshold_sample(pred_original_sample)
    elif self.config.clip_sample:
        pred_original_sample = pred_original_sample.clamp(
            -self.config.clip_sample_range, self.config.clip_sample_range
        )

    # 4. Compute coefficients for pred_original_sample x_0 and current sample x_t
    # See formula (7) from https://arxiv.org/pdf/2006.11239.pdf
    pred_original_sample_coeff = (alpha_prod_t_prev ** (0.5) * current_beta_t) / beta_prod_t
    current_sample_coeff = current_alpha_t ** (0.5) * beta_prod_t_prev / beta_prod_t

    # 5. Compute predicted previous sample µ_t
    # See formula (7) from https://arxiv.org/pdf/2006.11239.pdf
    pred_prev_sample = pred_original_sample_coeff * pred_original_sample + current_sample_coeff * sample

    # 6. Add noise
    if self.variance_type == "fixed_small_log":
        variance = _get_variance(self, t, prev_t, predicted_variance=predicted_variance)
        variance = unsqueeze_like(variance, variance_noise).to(variance_noise.dtype) * variance_noise
    elif self.variance_type == "learned_range":
        variance = _get_variance(self, t, prev_t, predicted_variance=predicted_variance)
        variance = torch.exp(0.5 * variance)
        variance = unsqueeze_like(variance, variance_noise).to(variance_noise.dtype) * variance_noise
    else:
        variance = (_get_variance(self, t, prev_t, predicted_variance=predicted_variance) ** 0.5)
        variance = unsqueeze_like(variance, variance_noise).to(variance_noise.dtype) * variance_noise
    variance[t==0] = 0.0

    pred_prev_sample = pred_prev_sample + variance

    if not return_dict:
        return (pred_prev_sample,)

    return DDPMSchedulerOutput(prev_sample=pred_prev_sample, pred_original_sample=pred_original_sample)


def ddpm_scheduler_step_to_orig_batched(self,
                                        model_output,
                                        timestep,
                                        sample):
    t = timestep

    if model_output.shape[1] == sample.shape[1] * 2 and self.variance_type in ["learned", "learned_range"]:
        model_output, predicted_variance = torch.split(model_output, sample.shape[1], dim=1)
    else:
        predicted_variance = None

    # 1. compute alphas, betas
    alpha_prod_t = self.alphas_cumprod[t]
    beta_prod_t = 1 - alpha_prod_t

    alpha_prod_t = unsqueeze_like(alpha_prod_t, sample).to(sample.dtype)
    beta_prod_t = unsqueeze_like(beta_prod_t, sample).to(sample.dtype)

    # 2. compute predicted original sample from predicted noise also called
    # "predicted x_0" of formula (15) from https://arxiv.org/pdf/2006.11239.pdf
    if self.config.prediction_type == "epsilon":
        pred_original_sample = (sample - beta_prod_t ** (0.5) * model_output) / alpha_prod_t ** (0.5)
        pred_original_noise = model_output
    elif self.config.prediction_type == "v_prediction":
        pred_original_sample = (alpha_prod_t**0.5) * sample - (beta_prod_t**0.5) * model_output
        pred_original_noise = (alpha_prod_t**0.5) * model_output + (beta_prod_t**0.5) * sample

    # 3. Clip or threshold "predicted x_0"
    if self.config.thresholding:
        pred_original_sample = self._threshold_sample(pred_original_sample)
    elif self.config.clip_sample:
        pred_original_sample = pred_original_sample.clamp(
            -self.config.clip_sample_range, self.config.clip_sample_range
        )
    return pred_original_sample, pred_original_noise