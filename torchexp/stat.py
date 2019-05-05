# Useful classes
class RunningAverage:
    def __init__(self, decay_rate, decay_step=1):
        self.r = decay_rate ** (1. / decay_step)
        self.z = 0.    # = 1 + r^2 + ... + r^{n-1}
        self.avg = 0.  # = 1/z sum_{i=1}^n r^{n-i} v_i

    def add(self, value, width=1):
        r_n = self.r ** width
        prev = self.z * r_n
        part = (1 - r_n) / (1 - self.r)
        self.z = prev + part
        self.avg = (self.avg * prev + value * part) / self.z

    def __float__(self):
        return self.avg

    def __str__(self):
        return f'{self.avg:.6f}'

    def __format__(self, spec):
        return format(self.avg, spec)


class F1Score:
    @staticmethod
    def compute(match, pred, label):
        '''
            match, pred, label: torch tensors with size (classes)
        '''
        c_r = (match / label.clamp(min=1.)).mean()
        c_p = (match / pred.clamp(min=1.)).mean()
        o_r = match.sum() / max(1., label.sum())
        o_p = match.sum() / max(1., pred.sum())
        return F1Score(c_r, c_p, o_r, o_p)

    def __init__(self, c_r=0., c_p=0., o_r=0., o_p=0.):
        self.c_r = c_r
        self.c_p = c_p
        self.o_r = o_r
        self.o_p = o_p
        self.c_f1 = 0.
        if c_r + c_p > 0:
            self.c_f1 = 2 * c_r * c_p / (c_r + c_p)

        self.o_f1 = 0.
        if o_r + o_p > 0:
            self.o_f1 = 2 * o_r * o_p / (o_r + o_p)

        self.avg_f1 = (self.c_f1 + self.o_f1) / 2

    def __str__(self):
        return (f'{self.c_r:.3f} {self.c_p:.3f} {self.c_f1:.3f} /'
                f'{self.o_r:.3f} {self.o_p:.3f} {self.o_f1:.3f}')

    def __format__(self, format_spec):
        return str(self)

    def verbose(self):
        return ('Per-class: '
                f'recall: {self.c_r:.3f} '
                f'precision: {self.c_p:.3f} '
                f'f1 score: {self.c_f1:.3f}\n'
                'Overall:   '
                f'recall: {self.o_r:.3f} '
                f'precision: {self.o_p:.3f} '
                f'f1 score: {self.o_f1:.3f}\n')

    def __lt__(self, score):
        return self.avg_f1 < score.avg_f1

    def __gt__(self, score):
        return self.avg_f1 > score.avg_f1
