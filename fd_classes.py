import numpy as np
import matplotlib.pylab as plt
import scipy.integrate as integrate


def gaussian(x, mu, sig):
    return np.exp(-0.5 * np.power((x - mu) / sig, 2)) / (sig * np.sqrt(2 * np.pi))


# def resample_data(data, binning, function=np.mean):
#     '''
#     Calculate bin wise funciton value of an array
#     Parameters
#     ----------
#     data: t, f(t) data
#     binning: The new binning
#     function: The Funciton to be applied
#
#     Returns
#     -------
#     tuple: Binning and Bin Mean Value
#
#     '''
#     bin_mean = []
#     if np.array_equal(data[0], binning):
#         return (binning, data[1])
#
#     for i in np.arange(len(binning) - 1):
#         mask = (binning[i] < data[0]) & (data[0] <= binning[i + 1])
#         values = data[1][mask]
#         if len(values) == 0:
#             print 'new_binning : no values in the range: %.2f - %.2f' % (binning[i], binning[i + 1])
#             bin_mean.append(np.nan)
#         else:
#             bin_mean.append(function(values))
#     return (np.array(binning), np.array(bin_mean))

# def resample_data(data, binning, function=np.mean):
#     '''
#     Calculate bin wise funciton value of an array
#
#     '''
#
#     digitized = np.digitize(data, binning)
#     bin_means = [function(data[digitized == i]) for i in range(1, len(binning))]
#
#     return (np.array(binning), np.array(bin_mean))

# def resample_data(data, binning, function=np.mean):
#     '''
#     Calculate bin wise funciton value of an array
#
#     '''
#
#     bin_mean = (np.histogram(data, binning, weights=data)[0] /
#                  np.histogram(data, binning)[0])
#
#     return (np.array(binning), np.array(bin_mean))

def resample_data(data, binning, function=np.mean):
    from scipy.stats import binned_statistic
    print 'resampling ...'

    if np.array_equal(data[0], binning):
        return (binning, data[1])

    bin_mean = binned_statistic(data[0], data[1], bins=binning, statistic=function)[0]

    return (np.array(binning), np.array(bin_mean))



def get_bin_mean(array):
    return (array[1:] + array[:-1]) / 2


class Wind_Data():
    '''
    Class for Wind Data analysis and plotting

    plot_properties: Dictionary of type {}
    '''

    def __init__(self, filename, sampling_frequency, plot_path=None, data=None, title='Data Set'):
        '''

        Parameters
        ----------
        filename: The filename of a one column data file
        sampling_frequency: Sampling Frequency in Hz
        data
        plot_properties
        title
        '''

        # DATA PROPERTIES
        self.filename = filename
        self.f = sampling_frequency
        self.plot_path = plot_path
        self.title = title
        self.data = None
        self.data_prime = None

        # ANALYSIS RESULTS
        self.dt = 1. / self.f
        self.t = None
        self.global_mean = None
        self.global_var = None
        self.global_std = None
        self.turbulance_intensity = None
        self.dissipation_rate = None

        # LOAD
        self.load_data()
        self.preprocess_data()
        self.calc_global_mean()
        self.calc_global_var()
        self.calc_global_std()
        self.calc_turbulance_intensity()
        self.calc_taylor_length()
        self.calc_integral_length()
        self.calc_kolmogorov_length()
        self.calc_data_prime()
        self.r = np.unique(np.array([int(1.6 ** i) for i in np.arange(26)]))

    def load_data(self):
        '''

        Returns
        -------
        Data Object
        '''
        # LOAD DATA FROM FILE
        self.data = np.genfromtxt(self.filename, dtype=float)
        # CREATE TIME ARRAY
        l = len(self.data)
        self.t = np.linspace(0, l * self.dt, num=l)
        print 'Sampling Frequency: %.2f Hz' % self.f
        print 'Length of Data Array: %d' % l
        print 'Total Measurement Time: %.f s' % (l * self.dt)

    def calc_global_mean(self):
        self.global_mean = np.nanmean(self.data)

    def calc_global_var(self):
        self.global_var = np.nanvar(self.data)

    def calc_global_std(self):
        self.global_std = np.nanstd(self.data)

    def calc_data_prime(self):
        self.data_prime = self.data - self.global_mean

    def calc_turbulance_intensity(self):
        if self.global_mean == None:  self.calc_global_mean()
        if self.global_var == None:  self.calc_global_var()

        self.turbulance_intensity = self.global_var / self.global_mean ** 2

    def calc_global_properties(self):
        self.calc_global_mean()
        self.calc_global_var()
        self.calc_turbulance_intensity()

    def preprocess_data(self):
        # REPLACE NAN VALUES BY GLOBAL MEAN VALUE
        if self.global_mean == None:
            self.calc_global_mean()
        self.data[self.data == np.nan] = self.global_mean

    def calc_power_spectrum(self, zero_padding=True):
        signal = self.data - self.global_mean
        N = len(signal)
        if not zero_padding:
            freq = np.fft.fftfreq(N, d=self.dt)
            fft = np.fft.fft(signal)
        elif zero_padding:
            freq = np.fft.fftfreq(2 * N - 1, d=self.dt)[0:N]
            fft = np.fft.fft(signal, 2 * N - 1)[0:N]
        return (freq, fft)

    def calc_autocorrelation(self):
        N = len(self.data)
        signal = self.data - self.global_mean
        d = N * np.ones(2 * N - 1)
        # acf = (np.correlate(signal, signal, 'full') / d)
        acf = (np.correlate(signal, signal, 'full') / d)
        acf = acf[N - 1:]
        acf_normalized = acf / self.global_var  # np.std(signal) ** 2
        step = np.arange(N)
        self.calc_integral_length(autocorr=(step, acf_normalized))
        self.calc_kolmogorov_length()

        return (step, acf_normalized)

    def calc_autocorrelation_fft(self, normalize=True, zero_padding=False):
        """
        Compute autocorrelation using FFT
        """
        signal = self.data
        x = signal - np.mean(signal)
        N = len(x)
        # remove mean value
        x = x - x.mean()
        # calculate fft, add zero padding
        if zero_padding:
            s = np.fft.fft(x , N*2-1)
            # calc ifft of Powerspectrum
            res = np.fft.ifft(np.abs(s) ** 2  , N*2-1)
        elif not zero_padding:
            s = np.fft.fft(x)
            # calc ifft of Powerspectrum
            res = np.fft.ifft(np.abs(s) ** 2)
        # take only the real part?????????????
        result = np.real(res)
        #print len(result)
        result = result[:N]
        #print len(result)
        # normalize
        if normalize:
            result /= result[0]
        return (self.dt , result)

    def calc_integral_length(self, autocorr=None):
        if autocorr == None:
            steps, acf_normalized = self.calc_autocorrelation()
        else:
            steps, acf_normalized = autocorr
        mask = acf_normalized > 0
        limit = np.where(acf_normalized < 0)[0][0]
        self.integral_length = integrate.simps(acf_normalized[:limit],
                                               self.t[:limit]) * self.global_mean  # , np.arange(N))

    def calc_kolmogorov_length(self):
        kinematic_visocsity_air = 1.42e-05  # m2/s
        if self.integral_length == None:  self.calc_integral_length()
        self.kolmogorov_length = (kinematic_visocsity_air ** 3 * self.integral_length / self.global_mean ** 3) ** (
            1. / 4.)
        # epsilon_1 = self.global_mean ** 3 / self.integral_length

    def calc_taylor_length(self):
        r = np.arange(10) + 1
        r2 = np.arange(11)

        taylor_length_2 = (r * self.dt * self.global_mean) ** 2 * np.mean(
            (self.data - self.data.mean()) ** 2) / self.calc_structure_function(2,
                                                                                r)
        res_2 = np.poly1d(np.polyfit(r, taylor_length_2, 2))

        self.taylor_length = np.sqrt(res_2(0))

    def calc_structure_function(self, degree, r, abs=True):
        if abs == True:
            return np.array([np.mean(np.abs(self.calc_diff(m)) ** degree) for m in r])
        result = []
        if abs == False:
            return np.array([np.mean(calc_diff(self.data, m) ** degree) for m in r])

    def calc_diff(self, n=1, fill=False):

        result = np.roll(self.data, - n)[:-n] - self.data[:-n]
        if fill:
            zeros = np.zeros(len(self.data) - len(result))
            result = np.insert(result, 0, zeros)
        return result

    def calc_scaling_S3(self, deg, S3_max):
        r = self.r
        Sn = self.calc_structure_function(deg, r, abs=True)
        S3 = self.calc_structure_function(3, r, abs=True)
        temp_mask = np.isfinite(S3) & (S3 < S3_max)
        scaling = np.polyfit(np.log10(S3[temp_mask]), np.log10(Sn[temp_mask]), 1)
        return scaling

    def calc_scaling_r(self, deg, l_min=None, l_max=None):
        if l_max == None: l_max = self.integral_length
        if l_min == None: l_min = self.taylor_length

        r = self.r
        l = r * self.dt * self.global_mean
        Sn = self.calc_structure_function(deg, r, abs=True)
        temp_mask = np.isfinite(l) & (l < l_max) & (l > l_min)
        scaling = np.polyfit(np.log10(l[temp_mask]), np.log10(Sn[temp_mask]), 1)
        return scaling

    def calc_probability_of_ur(self, r, normalize_to=None, normalize_to_std=False, bins=100, bin_edges=None, norm=True):

        diff = self.calc_diff(r)
        if normalize_to:
            diff /= normalize_to
        if normalize_to_std:
            diff /= np.std(diff)
        if bin_edges is not None:
            bins = bin_edges
            bin_val, edges = np.histogram(diff, bins=bins, normed=False)
            width = edges[1:] - edges[:-1]
        elif bin_edges is None:
            bin_val, edges = np.histogram(diff, bins=bins, normed=False)
            width = (edges.max() - edges.min()) / bins
        if norm:
            bin_val = bin_val.astype(np.float64) / np.sum(bin_val * width)
        return (bin_val, edges)

    def calc_pdf_cond1(self, lag_0, lag_1, projection_lag_1=50, bins=100):
        u_1 = self.calc_diff(n=lag_1, fill=True)  # [lag_2-1:]
        u_1 = u_1 / self.global_std

        # y axis
        u_0 = self.calc_diff(n=lag_0, fill=True)  # [lag_1-1:]
        u_0 = u_0 / self.global_std

        # create histogram
        x0 = projection_lag_1
        binning = np.linspace(-5, 5, bins)
        H, xedges, yedges = np.histogram2d(u_1, u_0, bins=binning, normed=True)
        p_x0 = H[:, x0] / sum(H[:, x0])
        L = yedges[1:] - yedges[:-1]
        return (get_bin_mean(yedges), p_x0 / L)

    def calc_pdf_cond2(self, lag_0, lag_1, lag_2, projection_lag_1=50, projection_lag_2=50, bins=100):
        # x axis
        u_1 = self.calc_diff(n=lag_1, fill=True)  # [lag_2-1:]
        u_1 = u_1 / self.global_std

        # y axis
        u_0 = self.calc_diff(n=lag_0, fill=True)  # [lag_1-1:]
        u_0 = u_0 / self.global_std

        # z axis
        u_2 = self.calc_diff(n=lag_2, fill=True)  # [lag_1-1:]
        u_2 = u_2 / self.global_std

        x_0 = projection_lag_1
        z_0 = projection_lag_2

        # create histogram
        binning = np.linspace(-5, 5, bins)
        H, edges = np.histogramdd((u_2, u_1, u_0), bins=(binning, binning, binning), normed=True)
        xedges, yedges, zedges = edges
        p_y0 = H[:, x_0, z_0] / sum(H[:, x_0, z_0])
        L = yedges[1:] - yedges[:-1]
        return (get_bin_mean(yedges), p_y0 / L)

    def plot_time_series(self, resampling_time=60 * 10, filename=None):
        from matplotlib.patches import Rectangle
        # data_frame_resampled_mean, data_frame_resampled_std = resample_data(frame)
        dt = self.dt
        t = self.t
        data = self.data
        binning = np.arange(0, dt * len(data), resampling_time)
        binning, mean = resample_data((t, data), binning, np.mean)
        binning, std = resample_data((t, data), binning, np.std)
        mean = np.array(mean)
        std = np.array(std)

        binning = (binning[1:] + binning[:-1]) / 2
        fig, ax = plt.subplots()
        ax.set_xlabel('Time / [s]')
        ax.set_title('MEAN: %.2f m/s, STD: %.2f m/s \n DOT: %.2e' % (
            data.mean(), data.std(), data.std() ** 2 / data.mean() ** 2))
        ax.set_ylabel('Wind Speed / [m/s]')
        ax.grid()
        ax.plot(t, data, label='measured data')
        local1, = ax.step(binning, mean, where='mid', color='green', label=r'$\bar{u}$')
        local2, = ax.step(binning, mean + std, where='mid', color='red', label=r'$\bar{u} \pm \sigma$')
        local3, = ax.step(binning, mean - std, where='mid', color='red', label=r'$\bar{u} - \sigma$')

        extra1 = Rectangle((0, 0), 1, 1, fc="w", fill=False, axes=ax, edgecolor='none', linewidth=0, label='local:')
        extra2 = Rectangle((0, 0), 1, 1, fc="w", fill=False, axes=ax, edgecolor='none', linewidth=0, label='global:')
        extra3 = Rectangle((0, 0), 1, 1, fc="w", fill=False, axes=ax, edgecolor='none', linewidth=0, label='DOT::')

        global1 = ax.axhline(y=data.mean(), label=r'$\bar{u}$')
        global2 = ax.axhline(y=data.mean() + data.std(), label='', color='black')
        global3 = ax.axhline(y=data.mean() - data.std(), color='black')

        # (r'$\bar{u}$', r'$\bar{u} + \sigma$', r'$\bar{u} - \sigma$')
        first_legend = ax.legend([extra1, local1, local2], ('local:', r'$\bar{u}$', r'$\bar{u} \pm \sigma$'),
                                 loc='upper center', bbox_to_anchor=(0.5, 1.0), ncol=4, fancybox=True, shadow=False)
        ax = plt.gca().add_artist(first_legend)
        plt.legend([extra2, global1, global2, extra3], ('global:', r'$\bar{u}$', r'$\bar{u} \pm \sigma$'),
                   loc='lower center', bbox_to_anchor=(0.5, 0.0), ncol=4, fancybox=True, shadow=False)

        # plt.legend(handles=[extra1], labels=('What is'), loc='lower center', bbox_to_anchor=(0.5, .0), ncol=3, fancybox=True, shadow=True)
        plt.show()
        if filename: fig.savefig(filename)

    def plot_pdf(self, mode='u', bar=False, interval=None, filename=None, fig_ax=None, bins=100, **kwargs):
        color = kwargs.pop('color','blue')
        alpha = kwargs.pop('alpha',1)
        label = kwargs.pop('label', None)
        if mode == 'u':
            data = self.data
            xlabel = 'u / [m/s]'
            ylabel = 'p(u)'
            binning = np.linspace(int(data.min()) - 1, int(data.max()) + 1, bins + 1)

        elif mode == 'u_prime':
            data = self.data_prime
            xlabel = "u\' / [m/s]"
            ylabel = "p(u')"
            binning = np.linspace(-3, 3, bins + 1)
        elif mode == 'u_prime_normalized':
            data = self.data_prime
            data /= self.global_std
            xlabel = r"u' / $\sigma_u$"
            ylabel = r"p(u' / $\sigma_u$)"
            binning = np.linspace(-3, 3, bins + 1)
        elif mode == 'interval':
            if interval:
                temp_binning = np.arange(0, self.dt * len(self.data), interval)
                if len(temp_binning) < 3:
                    print 'choose other interval'
                    return 1
                temp_binning, data = resample_data((self.t, self.data), temp_binning, np.mean)
                data -= np.nanmean(data)
                data /= np.nanstd(data)
                # print temp_binning
                # print data
                xlabel = r"u' / $\sigma_u$"
                ylabel = r"p(u' / $\sigma_u$)"
                binning = np.linspace(-3, 3, bins + 1)
                label = '%.1e s' % interval

        else:
            print 'MODE ERROR'

        if not fig_ax:
            fig = plt.figure()
            ax = fig.add_subplot(111)
        else:
            fig = fig_ax[0]
            ax = fig_ax[1]

        if mode == 'u_prime_normalized':
            ax.plot(np.linspace(-3, 3, 100), gaussian(np.linspace(-3, 3, 100), 0, 1), color='k', label='Gaussian')

        # ax.plot(np.linspace(-5, 5, 100), gaussian(np.linspace(-5, 5, 100), 0, 1), color='k', label='Gaussian')
        title = 'PDF'
        bin_val, edges = np.histogram(data, bins=binning, normed=False)
        bin_val_old = bin_val.copy()
        width = (edges.max() - edges.min()) / bins
        width = edges[1] - edges[0]
        #print bin_val
        bin_val = bin_val.astype(np.float64) / np.sum(bin_val * width)  # normalize

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_xlim(binning[0], binning[-1])
        res= None
        if bar:
            res= ax.bar((edges[:-1] + 0.5 * (edges[0] - edges[1])), bin_val, width=width,
                   label=label, color=color, alpha=alpha)
        elif not bar:
            ax.plot((edges[:-1] + 0.5 * (edges[0] - edges[1])), bin_val, marker= '.',
                    label=label)

        ax.grid()
        ax.legend()

        if filename:    fig.savefig(self.plot_path + filename)
        return ax

    def plot_time_dependent_pdf(self, intervals, filename=None):
        fig, ax1 = plt.subplots(1, sharex=False, sharey=False)
        plt.rcParams['axes.prop_cycle'] = ("cycler('color', ['r', 'g', 'b', 'y']) ")
                                           #"cycler('lw', [1, 2, 3])")
        for interval in intervals:
            self.plot_pdf(mode='interval', bar=False, interval=interval, filename=None, fig_ax=(fig, ax1), bins=30)
        ax1.grid()
        if filename:    fig.savefig(self.plot_path + filename)

    def plot_pdf_all(self, filename=None):
        fig, (ax1, ax2, ax3) = plt.subplots(3, sharex=False, sharey=False)
        #fig.tight_layout()
        fig.subplots_adjust(hspace=.5)
        #ax3.set_title('     u\' / [m/s]')
        #ax2.set_title('     u / [m/s]')

        ax1 = self.plot_pdf(mode='u', bar=True, filename=None, fig_ax=(fig, ax1))
        ax2=self.plot_pdf(mode='u_prime', bar=True, filename=None, fig_ax=(fig, ax2))
        ax3=self.plot_pdf(mode='u_prime_normalized', bar=True, filename=None, fig_ax=(fig, ax3))

        plt.show()

        if filename:    fig.savefig(self.plot_path + filename)

    def plot_pdf_all_2(self, filename=None):
        fig, ax1 = plt.subplots(1, sharex=False, sharey=False)
        ax2 = ax1.twiny()
        res1 = self.plot_pdf(mode='u', bar=True, filename=None, fig_ax=(fig, ax1),color='red', alpha=0.5, label='u')

        #self.plot_pdf(mode='u_prime', bar=True, filename=None, fig_ax=(fig, ax2))
        res2 = self.plot_pdf(mode='u_prime_normalized', bar=True, filename=None, fig_ax=(fig, ax2),color='blue', alpha=0.5, label=r"u'/ $\sigma$")

        plt.legend([res1, res2], ('u', r"u'/ $\sigma$"))
                   #loc='lower center', bbox_to_anchor=(0.5, 0.0), ncol=4, fancybox=True, shadow=False)
        plt.show()

        if filename:    fig.savefig(self.plot_path + filename)

    def plot_mean_value(self, interval=1, fig_ax=None, filename=None):
        '''

        Parameters
        ----------
        interval: averaging Interval
        fig_ax: figure, AXES Objects

        Returns
        -------

        '''
        binning = np.arange(0, self.dt * len(self.data), interval)

        if len(binning) < 3:
            print 'choose other interval'
            return 1
        binning, mean = resample_data((self.t, self.data), binning, np.mean)
        binning_mid = get_bin_mean(binning)

        if not fig_ax:
            fig = plt.figure()
            ax = fig.add_subplot(111)
        else:
            fig = fig_ax[0]
            ax = fig_ax[1]

        ax.set_xlabel('Time / [s]')
        ax.set_title('Time Dependent Mean Value')
        ax.set_ylabel('Mean / [m/s]')
        ax.grid()
        # ax.plot(binning_mid, mean, label='Mean')
        a, = ax.step(binning_mid, mean, where='mid', marker='o', label='I=%.3f s' % interval)
        ax.step(binning[:-1], mean[:], where='post', color=a.get_color())
        ax.step(binning[1:], mean[:], where='pre', color=a.get_color())
        # ax.step(binning[:-1], mean[:], where='post', label='I=%d s' % interval)
        ax.legend(loc='best')
        if filename:    fig.savefig(self.plot_path + filename)

    def plot_variance(self, interval=1, fig_ax=None):
        '''

        Parameters
        ----------
        interval: averaging Interval
        fig_ax: figure, AXES Objects

        Returns
        -------

        '''
        binning = np.arange(0, self.dt * len(self.data), interval)

        if len(binning) < 3:
            print 'choose other interval'
            return 1
        binning, mean = resample_data((self.t, self.data), binning, np.nanvar)
        binning_mid = get_bin_mean(binning)

        if not fig_ax:
            fig = plt.figure()
            ax = fig.add_subplot(111)
        else:
            fig = fig_ax[0]
            ax = fig_ax[1]

        ax.set_xlabel('Time / [s]')
        ax.set_title('Time Dependent Variance')
        ax.set_ylabel(r'$\sigma^2$ / [m/s]^2')
        ax.grid()
        # ax.plot(binning_mid, mean, label='Mean')
        a, = ax.step(binning_mid, mean, where='mid', marker='o', label='I=%.3f s' % interval)
        ax.step(binning[:-1], mean[:], where='post', color=a.get_color())
        ax.step(binning[1:], mean[:], where='pre', color=a.get_color())
        # ax.step(binning[:-1], mean[:], where='post', label='I=%d s' % interval)
        ax.legend(loc='best')

    def plot_DOT(self, interval=1, fig_ax=None):
        '''

        Parameters
        ----------
        interval: averaging Interval
        fig_ax: figure, AXES Objects

        Returns
        -------

        '''
        binning = np.arange(0, self.dt * len(self.data), interval)

        if len(binning) < 3:
            print 'choose other interval'
            return 1

        binning, mean = resample_data((self.t, self.data), binning, np.nanmean)
        binning, var = resample_data((self.t, self.data), binning, np.nanvar)

        turbulance_intensity = var / mean ** 2
        binning_mid = get_bin_mean(binning)

        if not fig_ax:
            fig = plt.figure()
            ax = fig.add_subplot(111)
        else:
            fig = fig_ax[0]
            ax = fig_ax[1]

        ax.set_xlabel('Time / [s]')
        ax.set_title('Time Dependent Density of Turbulance')
        ax.set_ylabel(r'DOT / [-]')
        ax.grid()
        # ax.plot(binning_mid, mean, label='Mean')
        a, = ax.step(binning_mid, turbulance_intensity, where='mid', marker='.', label='I=%.3f s' % interval)
        ax.step(binning[:-1], turbulance_intensity[:], where='post', color=a.get_color())
        ax.step(binning[1:], turbulance_intensity[:], where='pre', color=a.get_color())
        # ax.step(binning[:-1], mean[:], where='post', label='I=%d s' % interval)
        ax.legend(loc='best')

    def plot_different_characteristics(self, function, intervals=[100, 1000, 5000], filename=None):
        fig, ax1 = plt.subplots(1, sharex=False, sharey=False)
        plt.rc('axes', color_cycle=['r', 'g', 'b', 'y'])
        for i in intervals:
            function(i, (fig, ax1))

        if filename:    fig.savefig(self.plot_path + filename)

    def plot_different_characteristics_multiplot(self, intervals=[100, 1000, 5000], filename=None):
        fig, (ax1, ax2, ax3) = plt.subplots(3, sharex=False, sharey=False)
        plt.rc('axes', color_cycle=['r', 'g', 'b', 'y'])
        plt.rc('legend', fontsize=8)
        axes = [ax1, ax2, ax3]
        functions = [self.plot_mean_value, self.plot_variance, self.plot_DOT]
        for ind, function in enumerate(functions):
            for i in intervals:
                function(i, (fig, axes[ind]))

        ax1.set_title('')
        ax2.set_title('')
        ax3.set_title('')

        if filename:    fig.savefig(self.plot_path + filename)

    def plot_time_dependence(self, filename=None):
        fig, (ax1, ax2) = plt.subplots(2, sharex=True, sharey=False)
        #intervals = np.linspace(10*self.dt, self.t[-1], 30)
        intervals = np.arange(10 * self.dt, self.t[-1], 500 * self.dt)
        #intervals = np.linspace(10*self.dt,self.t[-1], self.dt)
        #print intervals
        DOT = []
        VAR = []
        for i in intervals:
            binning = np.arange(0, self.dt * len(self.data), i)
            binning, mean = resample_data((self.t, self.data), binning, np.mean)
            binning, var = resample_data((self.t, self.data), binning, np.var)
            turbulance_intensity = var / mean ** 2
            DOT.append(np.mean(turbulance_intensity))
            VAR.append(np.mean(var))
        ax1.plot(intervals, VAR, '-o')
        ax2.plot(intervals, np.array(DOT),'-o')
        ax1.set_ylabel(r'$\sigma^2$ / [m/s]^2')
        ax2.set_ylabel(r'DOT / [-]')
        ax2.set_xlabel('Length of interval / [s]')
        ax1.set_title('Dependence on the averaging length')
        ax1.grid()
        ax2.grid()
        if filename:    fig.savefig(self.plot_path + filename)

    def plot_power_spectrum(self, bins=100, filename=None, const=1):
        freq, fft = self.calc_power_spectrum()
        N = len(self.data)
        binning = np.logspace(np.floor(np.log10(np.abs(freq[1]))), np.floor(np.log10(np.abs(freq[-1]))), num=bins)
        binning, bin_mean = resample_data((freq[1:N / 2], 2 * np.abs(fft[1:N / 2]) ** 2), binning)
        bin_mid = (binning[:-1] + binning[1:]) / 2
        bin_mean = np.array(bin_mean)
        fig, ax = plt.subplots()
        ax.grid()
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.plot(freq[1:N / 2], 2 * np.abs(fft[1:N / 2]) ** 2, color='red', marker='.')
        ax.plot(bin_mid, bin_mean, marker='.', label='Mean Values', color='blue')
        max = np.log10(np.nanmax(bin_mean)) - np.abs(np.log10(np.nanmin(freq[1:N / 2])))
        ax.plot(freq[1:N / 2], 10 ** (-5 / 3 * np.log10(freq[1:N / 2])) * int(max)*const, label='-5/3 scaling', color='green')
        ax.set_xlabel('Frequency / [Hz]')
        ax.set_ylabel('PSD / [a.u]')
        ax.set_title('Power Spectrum')
        ax.legend(loc='best')
        if filename:    fig.savefig(self.plot_path + filename)
        plt.show()

    def plot_autocorrelation(self, filename=None):
        fig, ax = plt.subplots()
        ax.set_title('Autocorrelation Function')
        ax.set_xlabel(r'$\tau$ / [m]')
        ax.set_ylabel(r'R$_{uu}$($\tau$)')
        ax.grid()
        steps, acf_normalized = self.calc_autocorrelation()
        # ax.plot(np.arange(N), calc_acf(data, unbiased=False, nlags=N, fft=False), color='blue', marker='.',alpha=0.5)
        ax.plot(steps * self.dt * self.global_mean, acf_normalized, color='red', marker='.', alpha=0.5)

        if self.integral_length == None:
            self.calc_integral_length()
        ax.axvline(x=self.integral_length, label='Integral Length')
        # if self.kolmogorov_length==None:
        #     self.calc_kolmogorov_length()

        # ax.axvline(x=self.kolmogorov_length, label='Kolmogorov Length')
        # ax.plot(np.arange(len(acf_normalized>0)), acf_normalized[acf_normalized>0], color='blue', marker='.')
        # fig.savefig(plotfolder + 'Autocorrelation.png')
        if filename:    fig.savefig(self.plot_path + filename)
        plt.legend()
        plt.show()

    def plot_taylor_length(self, filename=None):

        r = np.arange(10) + 1
        r2 = np.arange(11)

        taylor_length_2 = (r * self.dt * self.global_mean) ** 2 * np.mean(
            (self.data - self.data.mean()) ** 2) / self.calc_structure_function(2,
                                                                                r)
        res_2 = np.poly1d(np.polyfit(r, taylor_length_2, 2))

        self.taylor_length = np.sqrt(res_2(0))

        fig, ax = plt.subplots()
        ax.grid()
        ax.set_title('Taylor Length')
        ax.set_ylabel(r'$\lambda^2$ / [m$^2$]')
        ax.set_xlabel(r'r (steps)')
        ax.plot(r, taylor_length_2, 'o', label='Taylor_length')
        # ax.plot(data.mean()*np.arange(10),res(np.arange(10))**2,label='fit')
        ax.plot(r2, res_2(r2), label='Interpolated 2nd Order')
        ax.plot(0, res_2(0), marker='o', label=r'$\lambda=%.2e$ m' % self.taylor_length)
        ax.legend(loc='best')
        # fig.savefig(plotfolder + 'Taylor_length.png')
        if filename:    fig.savefig(self.plot_path + filename)
        plt.show()

    def plot_velocity_increment(self, filename=None):
        steps = 26  # 14
        # r = np.unique(np.array([int(1.6 ** i) for i in np.arange(steps)]))
        # r = np.array([1, 2 ,4 ,8])
        # u_incr = np.array([self_diff(data, n=m) for m in r])


        fig, ax = plt.subplots()
        ax.grid()
        ax.set_title('Velocity Increment')
        ax.set_xlabel('t / [s]')
        ax.set_ylabel(r'u_r / [m/s]')
        for i in [32,16,8,4,2,1]:
            ax.plot(self.t[i:], self.calc_diff(n=i), label='r = %d' % (i),
                    marker='.', alpha=0.5)
        # ax.plot(self.t[100:], self.calc_diff(n=100), label='lag = %.3e s' % (self.dt * 100), color='red', marker='.',
        #         alpha=0.5)
        # ax.plot(self.t[1:], self.calc_diff(n=1), label='lag = %.3e s' % (self.dt * 1), color='blue', marker='.',
        #         alpha=0.5)
        ax.legend(loc='best')
        # fig.savefig(plotfolder + 'velocity_increment.png')
        if filename:    fig.savefig(self.plot_path + filename)
        plt.show()

    def plot_structure_function(self, filename=None):
        steps = 26  # 14
        r = np.unique(np.array([int(1.6 ** i) for i in np.arange(steps)]))

        l = r * self.dt * self.global_mean

        # l_min = np.where(l > self.taylor_length)[0][0]
        # l_max = np.where(l < self.integral_length)[0][-1]

        S = [self.calc_structure_function(deg, r, abs=True) for deg in np.arange(7) + 2]

        fig, ax = plt.subplots()
        ax.grid()
        ax.set_title(r'Higher Order Structure Function')
        ax.set_xlabel('length / [m]')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_ylabel(r'$S_r^n$ / [m/s]$^n$')
        for ind, val in enumerate(S):
            ax.plot(l[:], val[:], marker='.', label='S%s' % (ind + 2))
        l1 = ax.legend(loc=4)

        v1 = plt.axvline(x=self.taylor_length, ls='dashed', color='black', label='Taylor Length')
        v2 = plt.axvline(x=self.integral_length, ls='dashed', color='black', label='Integral Length')

        l2 = ax.legend(handles=[v1, v2], loc=2)
        # add l1 again
        ax.add_artist(l1)
        # fig.savefig(plotfolder
        # + 'structure_function_higher.png')
        if filename:    fig.savefig(self.plot_path + filename)
        plt.show()

    def plot_scaling_S3(self, filename=None):
        steps = 26  # 14
        r = np.unique(np.array([int(1.6 ** i) for i in np.arange(steps)]))

        scaling_list = []

        fig, ax = plt.subplots()
        ax.grid()
        ax.set_title(r'ESS Scaling')
        ax.set_xlabel('S3')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_ylabel(r'$S_r^n$ / [m/s]$^n$')

        S3 = self.calc_structure_function(3, r, abs=True)

        for deg in [2, 3, 4, 5, 6, 7, 8]:
            Sn = self.calc_structure_function(deg, r, abs=True)

            temp_mask = np.isfinite(S3) & (S3 < 1)
            scaling = self.calc_scaling_S3(deg, 1)
            scaling_list.append(scaling[0])

            p, = ax.plot(S3, Sn, 'o')
            co = p.get_color()
            ax.plot(S3[temp_mask], 10 ** (scaling[1] + scaling[0] * np.log10(S3[temp_mask])), color=co,
                    label='S%s Fit m=%.2f' % (deg, scaling[0]))
            # ax.plot()

        ax.legend(loc='best')
        if filename:    fig.savefig(self.plot_path + '_1_'+ filename)
        # fig.savefig(plotfolder + 'structure_function_higher_scale2.png')

        fig, ax = plt.subplots()
        ax.grid()
        ax.set_title(r'Scaling of Sn as a Function of S3')
        ax.set_xlabel('n')
        ax.set_ylabel(r'Scaling')

        n = [2, 3, 4, 5, 6, 7, 8]
        fit = np.polyfit(n, scaling_list, 1)
        scaling = np.poly1d(fit)

        ax.plot(n, scaling_list, 'ro')
        ax.plot(n, scaling(n), 'b-', label='Fit: m=%.2f' % fit[0])

        n=np.array(n).astype(float)

        shiftK41 = scaling(n[0]) - n[0]/3.
        ax.plot(n, n/3. + shiftK41, label='K41')
        ax.plot(n, n/3. + (3.*n-n**2.)*0.25 / 18. + shiftK41, label='K62')
        ax.plot(n, n/9. + 2. - 2. * (2./3.)**(n/3.)+ shiftK41, label='SuL')

        ax.legend(loc='best')
        if filename:    fig.savefig(self.plot_path + '_2_' + filename)
        # fig.savefig(plotfolder + 'structure_function_higher_scale2_1.png')
        plt.show()

    def plot_Khinchin(self,filename=None):
        a = self.calc_autocorrelation()
        b = self.calc_autocorrelation_fft(zero_padding=False)
        c = self.calc_autocorrelation_fft(zero_padding=True)

        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()

        ax1.grid()
        ax1.set_title(u'Comparison of Different Methods \n to Determine Autocorrelation ')
        ax1.set_xlabel(r'$\tau$ / [m]')
        ax1.set_ylabel(r'R$_{uu}$($\tau$)')

        ac, =ax2.plot(a[1] - c[1], label='Formula - FFT with padding', alpha=0.2, color='grey')


        ax1.plot(b[1], label='FFT', color='green')
        ax1.plot(c[1], marker='.',label='FFT padding', color='red', alpha=0.5)
        ax1.plot(a[1], marker='+', label='Formula', color='blue', alpha=0.7)

        ax1.legend(loc='best')
        ax2.legend(loc='best')
        plt.show()

        if filename:    fig.savefig(self.plot_path + filename)

        return (fig,ax1,ax2)




    def plot_scaling_r(self, l_min=None, l_max=None, filename=None):
        if not l_max:   l_max= self.integral_length
        if not l_min:   l_min = self.taylor_length

        steps = 26  # 14
        r = np.unique(np.array([int(1.6 ** i) for i in np.arange(steps)]))
        l = r * self.dt * self.global_mean
        scaling_list = []

        fig, ax = plt.subplots()
        ax.grid()
        ax.set_title(r'Higher Order Structure Function - Scaling 2')
        ax.set_xlabel('r')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_ylabel(r'$S_r^n$ / [m/s]$^n$')

        for deg in [2, 3, 4, 5, 6, 7, 8]:
            Sn = self.calc_structure_function(deg, r, abs=True)

            temp_mask = np.isfinite(l) & (l > l_min) & (l < l_max)
            scaling = self.calc_scaling_r(deg,l_min=l_min,l_max=l_max)
            scaling_list.append(scaling[0])

            p, = ax.plot(l, Sn, 'o')
            co = p.get_color()
            ax.plot(l[temp_mask], 10 ** (scaling[1] + scaling[0] * np.log10(l[temp_mask])), color=co,
                    label='S%s Fit m=%.2f' % (deg, scaling[0]))
            # ax.plot()

        l1 = ax.legend(loc=4)
        v1 = plt.axvline(x=self.taylor_length, ls='dashed', color='black', label='Taylor Length')
        v2 = plt.axvline(x=self.integral_length, ls='dashed', color='black', label='Integral Length')
        l2 = ax.legend(handles=[v1, v2], loc=2)
        # add l1 again
        ax.add_artist(l1)
        if filename:    fig.savefig(self.plot_path + '_1_' + filename)
        plt.show()
        # fig.savefig(plotfolder + 'structure_function_higher_scale2.png')


        fig, ax = plt.subplots()
        ax.grid()
        ax.set_title(r'Scaling of S3 as a Function of Sn')
        ax.set_xlabel('n')
        ax.set_ylabel(r'Scaling')

        n = [2, 3, 4, 5, 6, 7, 8]
        fit = np.polyfit(n, scaling_list, 1)
        scaling = np.poly1d(fit)

        ax.plot(n, scaling_list, 'ro')
        ax.plot(n, scaling(n), 'b-', label='Fit: m=%.2f' % fit[0])
        n=np.array(n).astype(float)

        shiftK41 = scaling(n[0]) - n[0]/3.
        ax.plot(n, n/3. + shiftK41, label='K41')
        ax.plot(n, n/3. + (3.*n-n**2.)*0.25 / 18. + shiftK41, label='K62')
        ax.plot(n, n/9. + 2. - 2. * (2./3.)**(n/3.)+ shiftK41, label='SuL')

        ax.legend(loc='best')
        if filename:    fig.savefig(self.plot_path + '_2_' + filename)
        plt.show()

    def plot_incremental_pdf(self, lags, log_scale=True, filename=None, bar=True, normalize=True):
        '''
        PLOTS THE 2 POINT STATISTICS
        :param lag: Determines how far the two curves need to be shifted, 1 equals 0.02 seconds
        :param save: Dictionary of structure save={'save': True, 'format': 'png','path': plot_folder}
        :param kwargs:
        :return:
        '''
        data = self.data
        data = data - np.nanmean(data)
        if normalize:
            binning = np.linspace(-7, 7, 101)
            bins = len(binning) - 1
        if not normalize:
            binning = 100
            bins = 100

        fig = plt.figure()
        ax = fig.add_subplot(111)
        if normalize:
            ax.plot(np.linspace(-7, 7, 100), gaussian(np.linspace(-7, 7, 100), 0, 1), color='k', label='Gaussian')
        for lag in lags:
            if lag == 0:
                diff = data - data.mean()
                title = 'PDF'

            elif lag > 0:
                diff = self.calc_diff(lag)
                title = 'Incremental PDF'


            if normalize:   diff /= np.std(diff)

            bin_val, edges = np.histogram(diff, bins=binning, normed=False)
            width = (edges.max() - edges.min()) / bins
            bin_val = bin_val.astype(np.float64) / np.sum(bin_val * width)  # normalize
            if normalize:
                ax.set_xlabel(r'u$_{\tau}$ / $\sigma_{\tau}$')
                ax.set_xlim(-7, 7)
                ax.set_ylabel(r'p(u$_{\tau}$ / $\sigma_{\tau})$')
            if not normalize:
                ax.set_xlabel(r'u$_{\tau}$')
                ax.set_ylabel(r'p(u$_{\tau})$')


            ax.grid()
            if log_scale:
                ax.set_yscale('log')
                bin_val[bin_val==0]=np.nan

            if bar:
                ax.bar((edges[:-1] + 0.5 * (edges[0] - edges[1])), bin_val, width=width,
                       label=r'$\tau$= %f s' % (self.dt * lag),
                       color='r')
            elif not bar:
                ax.plot((edges[:-1] + 0.5 * (edges[0] - edges[1])), bin_val,
                        label=r'$\tau$= %f s' % (self.dt * lag))

            plt.title(title)
            ax.legend(loc='best')

        if filename:    fig.savefig(self.plot_path + filename)
        plt.show()

    def plot_joint_probabilities(self, tau=1, bins=100, filename = None):
        #plot ustrich
        shift = np.int(tau/self.dt)
        # x axis
        u=self.data_prime[:-shift]


        # y axis
        u_tau = self.data_prime[shift:]

        #u_0 = np.pad(u_1[shift:], (shift, 0), 'constant', constant_values=(0))
        #u_0= u_1[shift:]
        # create histogram
        binning = np.linspace(u.min(), u.max(), bins)

        H, xedges, yedges = np.histogram2d(u_tau, u, bins=bins, normed=True)

        # for Histo plotting
        x = (xedges[:-1] + xedges[1:]) / 2
        y = (xedges[:-1] + xedges[1:]) / 2
        A_bin = (xedges[1] - xedges[0]) * (yedges[1] - yedges[0])

        # PLOTTING
        fig, ax = plt.subplots()
        ax.set_xlabel(r"u'(t) / [m/s]")
        ax.set_ylabel(r"u'(t+%d s) / [m/s]"%tau)
        #ax.set_ylabel(r'$u_{%d} / std $' % lag0)
        ax.set_title('Multipoint Statistics')
        cmap = plt.cm.jet

        #### im = plt.imshow(x,y, H, interpolation='none', origin='low')
        im = ax.pcolormesh(x, y, H, cmap=cmap)#, vmin=0,vmax=1)
        #levels = np.logspace(-5, np.log10(H.max()), 10)
        levels = np.linspace(0,H.max(),10)
        CS = plt.contour(x, y, H,  cmap=plt.cm.gray, levels=levels)
        plt.clabel(CS, inline=1, fontsize=10)
        cbar = fig.colorbar(im)
        cbar.set_label(r"p(u'(t),u'(t+%d s))"%tau)
        plt.legend(loc='best')
        plt.show()
        if filename: fig.savefig(self.plot_path + filename)


    def N_point(self, lag0, lag1, projection_lag=50, bins=100):

        # x axis
        u_1 = self.calc_diff(n=lag1, fill=True)  # [lag_2-1:]
        u_1 = u_1 / self.global_std

        # y axis
        u_0 = self.calc_diff(n=lag0, fill=True)  # [lag_1-1:]
        u_0 = u_0 / self.global_std

        # create histogram
        binning = np.linspace(-5, 5, bins)
        H, xedges, yedges = np.histogram2d(u_1, u_0, bins=binning, normed=True)

        # for Histo plotting
        x = (xedges[:-1] + xedges[1:]) / 2
        y = (xedges[:-1] + xedges[1:]) / 2
        A_bin = (xedges[1] - xedges[0]) * (yedges[1] - yedges[0])

        # PLOTTING
        fig, ax = plt.subplots()
        ax.set_xlabel(r'$u_{%d}$ / std' % lag1)
        ax.set_ylabel(r'$u_{%d} / std $' % lag0)
        ax.set_title('Multipoint Statistics')
        cmap = plt.cm.jet

        #### im = plt.imshow(x,y, H, interpolation='none', origin='low')
        # im = ax.pcolormesh(x, y, H, cmap=cmap)
        levels = np.logspace(-5, np.log10(H.max()), 10)
        CS = plt.contour(x, y, H, levels=levels, cmap=cmap)
        plt.clabel(CS, inline=1, fontsize=10)
        cbar = fig.colorbar(CS)
        cbar.set_label(r'p($u_r$)')
        plt.legend(loc='best')
        # fig.savefig(plotfolder + 'cond_2D.png')
        # look for specific u=0
        x0 = projection_lag

        # 1D Projection
        fig, ax = plt.subplots()
        ax.set_yscale('log')
        ax.set_xlabel(r'u_1 / [$\sigma$]: lag%.2f' % lag0)
        ax.set_ylabel('pdf')

        p_r_val, p_r_edges = self.calc_probability_of_ur(lag0, normalize_to=self.global_std, bin_edges=binning)
        p_r_bin_mean = get_bin_mean(p_r_edges)
        x_0_value = p_r_bin_mean[x0]
        L1 = p_r_bin_mean[1] - p_r_bin_mean[0]
        #print L1
        L2 = L1
        ax.plot(p_r_bin_mean, p_r_val, label='p($u_{%.2f}$)' % lag0, marker='+')

        p_x0 = H[:, x0] / sum(H[:, x0])
        ax.plot(y, p_x0 / L2,
                label=r'p($u_{%.2f}$| $u_{%.2f}$=%.2f)' % (lag0, lag1, x_0_value))  # u'cond u1|u2') #ODER L2??????
        ax.legend()
        # fig.savefig(plotfolder + 'cond_1D.png')
        plt.show()
        #print L1, L2
        return (y, p_x0 / L2)


    def plot_N_point(self, lag0, lag1, projection_lag=50, bins=100, filename=None):

        # x axis tau_0
        u_x = self.calc_diff(n=lag0, fill=True)  # [lag_2-1:]

        # y axis tau_1
        u_y= self.calc_diff(n=lag1, fill=True)  # [lag_1-1:]

        # joint probabilites
        min=u_x.min().astype(int)
        binning = np.linspace(min, -min, bins)
        #binning = np.linspace(-3, 3, bins)
        H, xedges, yedges = np.histogram2d(u_x, u_y, bins=binning, normed=True)
        H = H.astype(np.float64)
        H /= np.nansum(H)

        x = (xedges[:-1] + xedges[1:]) / 2
        y = (xedges[:-1] + xedges[1:]) / 2
        bin_width = binning[1] - binning[0]

        # probability of condition
        p_u_1, temp = np.histogram(u_y, bins=binning, density=False)
        p_u_1 = p_u_1.astype(float)
        p_u_1 /= np.nansum(p_u_1)

        # conditional probabilities
        # H=H.transpose()
        H /= p_u_1

        fig, ax = plt.subplots()
        ax.set_xlabel(r'$u_{%d}$ / std' % lag1)
        ax.set_ylabel(r'$u_{%d} / std $' % lag0)
        ax.set_title('Multipoint Statistics')

        cmap = plt.cm.jet
        #im = ax.pcolormesh(x, y, H, cmap=cmap)
        levels = np.logspace(-3, np.log10(np.nanmax(H[H<100])), 15)
        levels = np.logspace(-6, 0, 12)
        CS = plt.contour(x, y, H, levels=levels, cmap=cmap)  # , cmap=cmap, levels=levels)
        cbar = fig.colorbar(CS)
        cbar.set_label(r'p($u_r$)')
        plt.legend(loc='best')

        if filename: fig.savefig(self.plot_path + filename)

    def plot_Markov_properties(self, lag1, lag2, lag3):
        fig, ax = plt.subplots()
        ax.set_yscale('log')
        ax.set_xlabel(r'u_1 / [$\sigma$]')
        ax.set_ylabel('pdf')
        ax.set_title('r1=%d, r2=%d, r3=%d' % (lag1, lag2, lag3))
        a = self.calc_pdf_cond1(lag1, lag2)
        b = self.calc_pdf_cond2(lag1, lag2, lag3)
        ax.plot(a[0], a[1], label=u'u1|u2')
        ax.plot(b[0], b[1], label=u'u1|u2,u3')
        ax.legend(loc='best')
        # fig.savefig(plotfolder + 'cond_comparison.png')
        plt.show()


#Data = Wind_Data('/home/jo/DATA/Fluid_Dynamic/AtmosphericData_July_fs10Hz_Kurz.txt', sampling_frequency=10,
#               plot_path='/home/jo/Dropbox/Fluiddynamik/Protokoll/PLOTS/ATMOSPHERE/')

#Data = Wind_Data('/home/jo/DATA/Fluid_Dynamic/Data_Centerline_FractalGrid_fs60kHz.txt', sampling_frequency=60000., plot_path ='/home/jo/Dropbox/Fluiddynamik/Protokoll/PLOTS/LAB/')
#Data.plot_joint_probabilities( tau=1, bins=50. filename='Joint_probabilities.png')

#Data.plot_time_dependence(filename= 'time_dependence.png')

# Data.load_data()
# Data.preprocess_data()
# Data.calc_global_properties()

# DIFFERENT BINNING - 10 MINUTE AVERAGE
# Data.plot_time_series(resampling_time=60*10)
# Data.plot_time_series(resampling_time=60)

#
# Data.plot_pdf(mode='u')
# Data.plot_pdf(mode='u_prime')
# Data.plot_pdf(mode='u_prime_normalized')

#Data.plot_pdf_all(filename = 'PDF_ALL.png')
#Data.plot_pdf_all_2(filename = 'PDF_ALL_2.png')

# Data.plot_mean_value(interval=60*10, fig_ax=None)
# Data.plot_different_characteristics(Data.plot_mean_value, intervals=[60,600, 5500], filename= None)
# Data.plot_different_characteristics(Data.plot_variance, intervals=[600,1500,5000], filename= None)
# Data.plot_different_characteristics(Data.plot_DOT, intervals=[600,1500,5000], filename= None)
#Data.plot_different_characteristics_multiplot(intervals=[10,600,2500, 5500], filename= 'Characteristics_multiplot.png')
# Data.plot_pdf(mode='interval', interval=1, bar=True, bins=30)

#Data.plot_time_dependence(filename= 'time_dependence.png')

#Data.plot_time_dependent_pdf(intervals=[0.2, 2, 20, 200],  filename= "time_dependence_pdf.png")
#Data.plot_time_dependent_pdf(intervals=[200], filename= 'time_dependence_pdf.png')

# POWER SPECTRUM
#Data.plot_power_spectrum(filename='Power_Spectrum.png' , const=70)

# AUTOCORRELATION
#Data.plot_autocorrelation(filename='Autocorrelation.png')

# TAYLOR LENGTH
#Data.plot_taylor_length(filename='Taylor_length.png')

# VELOCITY INCREMENT
#Data.plot_velocity_increment(filename= 'velocity_increment.png')

# STRUCTURE FUNCTION
#Data.plot_structure_function(filename='structure_function_higher.png')

# SCALING 2 (S)
#Data.plot_scaling_S3(filename='S3_scaling.png')

#Data.plot_scaling_r(r_max=200, filename='r_scaling.png')

# 2 POINT STATS
#Data.plot_incremental_pdf([1,100,1000], bar=False)

# N Point Statisitics
#Data.N_point(44 , 88, projection_lag=50 , bins=100)

# Data.plot_Markov_properties(44, 88, 132)
