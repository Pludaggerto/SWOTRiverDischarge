from Base import Util
import pymc3 as pm
import logging
import numpy as np

class BAM(object):
    
    def __init__(self):
        self.BAM = Util.read_Sacramento()
        self.w = self.BAM["Sac_w"]
        self.s = self.BAM["Sac_s"]
        self.dA = self.BAM["Sac_dA"]
        self.QWBM = self.BAM["Sac_QWBM"]
        self.Qobs = self.BAM["Sac_Qobs"]

    def __del__(self):
        pass

    def plot_data(self):
        pass

    def run(self):
        # prior Qt
        Qmax = self.w.max() * 5 * 10
        Qmin = self.w.min() * 0.5 * 0.5

        # prior logA0i
        # mulA0i = 0.855 + 1.393 * np.log(self.w).mean() - 1.7249043 * np.log(self.w).var()

        # prior Qc, Wc, bi
        wmean = self.w.mean()
        logwvar = np.log(self.w).var()
        mubi = 0.02161 + 0.4578 * logwvar

        with pm.Model() as BAM_model:

            Qt = pm.TruncatedNormal("Qt", mu = self.QWBM, sigma = 1, lower = Qmin, upper = Qmax)
            # A0i = pm.Lognormal("A0i", mu = mulA0i, sigma = 0.948)
            #n = pm.Lognormal("n", mu = -3.5, sigma = 1)
            Qc = pm.TruncatedNormal("Qc", mu = self.QWBM, sigma = 1, lower = Qmin, upper = Qmax)
            Wc = pm.Lognormal("Wc", mu = wmean, sigma = 0.01)
            bi = pm.Lognormal("bi", mu = mubi, sigma = 0.098)
            sigma_g = pm.Normal("sigma_g", mu = 0)
            Wmu = bi * (np.log(Qt) - np.log(Qc)) + Wc
            Wobser = pm.Lognormal("Wobser", mu = Wmu, sd = sigma_g, observed = self.w)
            
            start = pm.find_MAP()
            trace = pm.sample(5000, return_inferencedata=True)
            pm.traceplot(trace)
            pm.summary(trace)
        
def main():
    
    log = logging.getLogger()
    handler = logging.StreamHandler()
    log.addHandler(handler)
    log.setLevel(logging.INFO)
    
    bam = BAM()
    bam.run()
     
if __name__ == '__main__':
    main()
