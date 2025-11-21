class EarlyStopper:
    def __init__(self, tol_upd_floor=2e-4, tol_rel_floor=5e-4,
                 early_frac=0.40, ema_alpha=0.5, patience=2, min_iters=3,
                 status_cb=lambda s: None):
        self.tol_upd_floor = float(tol_upd_floor)
        self.tol_rel_floor = float(tol_rel_floor)
        self.early_frac    = float(early_frac)
        self.ema_alpha     = float(ema_alpha)
        self.patience      = int(patience)
        self.min_iters     = int(min_iters)
        self.status_cb     = status_cb

        self.ema_um = None
        self.ema_rc = None
        self.base_um = None
        self.base_rc = None
        self.early_cnt = 0

        self.status_cb(
            "MFDeconv early-stop config: "
            f"tol_upd_floor={self.tol_upd_floor:g}, tol_rel_floor={self.tol_rel_floor:g}, "
            f"early_frac={self.early_frac:g}, ema_alpha={self.ema_alpha:g}, "
            f"patience={self.patience}, min_iters={self.min_iters}"
        )

    def step(self, it, max_iters, um, rc):
        um = float(um); rc = float(rc)

        if it == 1 or self.ema_um is None:
            self.ema_um = um
            self.ema_rc = rc
            self.base_um = um
            self.base_rc = rc
        else:
            a = self.ema_alpha
            self.ema_um = a * um + (1.0 - a) * self.ema_um
            self.ema_rc = a * rc + (1.0 - a) * self.ema_rc

        b_um = self.base_um if (self.base_um and self.base_um > 0) else um
        b_rc = self.base_rc if (self.base_rc and self.base_rc > 0) else rc

        tol_um = max(self.tol_upd_floor, self.early_frac * b_um)
        tol_rc = max(self.tol_rel_floor, self.early_frac * b_rc)

        small = (self.ema_um < tol_um) or (self.ema_rc < tol_rc)

        self.status_cb(
            f"MFDeconv iter {it}/{max_iters}: "
            f"um={um:.3e}, rc={rc:.3e} | "
            f"ema_um={self.ema_um:.3e}, ema_rc={self.ema_rc:.3e} | "
            f"tol_um={tol_um:.3e}, tol_rc={tol_rc:.3e} | small={bool(small)}"
        )

        if small and it >= self.min_iters:
            self.early_cnt += 1
            self.status_cb(f"MFDeconv iter {it}: early-stop candidate ({self.early_cnt}/{self.patience})")
        else:
            if self.early_cnt:
                self.status_cb(f"MFDeconv iter {it}: early-stop streak reset")
            self.early_cnt = 0

        if self.early_cnt >= self.patience:
            self.status_cb(
                "MFDeconv early-stop TRIGGERED: "
                f"ema_um={self.ema_um:.3e} < {tol_um:.3e} or "
                f"ema_rc={self.ema_rc:.3e} < {tol_rc:.3e} "
                f"for {self.patience} consecutive iters"
            )
            return True  # stop now

        return False
