class OptimiserFailed(Exception):
    pass


class Scheduler:
    def __init__(self, optimiser, lr_step, lr_min, patience=10, rtol=0.25, atol=0.1, eps=1e-2):
        self.optimiser = optimiser
        self._steps = 0
        self.top_obj = float('inf')

        self._rtol = rtol
        self._atol = atol

        self.patience = patience
        self._patience = patience
        self._lr_step = lr_step
        self._lr_min = lr_min
        self._eps = eps
        self.is_best = False

    def step(self, objective):
        self._steps += 1

        if objective < self.top_obj - self._eps:
            self.top_obj = objective
            self.is_best = True
            self._patience = self.patience
        else:
            self.is_best = False
            if objective > self.top_obj * (1 + self._rtol) + self._atol:
                raise OptimiserFailed()
            else:
                self._patience -= 1

        if self._patience == 0:
            self.lr *= self._lr_step
            self._patience = self.patience

    def step_lr(self):
        self.lr = self._lr_step(self.lr)

    @property
    def lr(self):
        for pg in self.optimiser.param_groups:
            lr = pg['lr']
        return lr

    @lr.setter
    def lr(self, new_lr):
        if new_lr < self._lr_min:
            pass
        else:
            for pg in self.optimiser.param_groups:
                pg['lr'] = new_lr

    def state_dict(self):
        return {'optim_state_dict': self.optimiser.state_dict(),
                'steps': self._steps,
                'top_obj': self.top_obj}

    def load_state_dict(self, state_dict):
        self.optimiser.load_state_dict(state_dict['optim_state_dict'])
        self._steps = state_dict['steps']
        self.top_obj = state_dict['top_obj']
