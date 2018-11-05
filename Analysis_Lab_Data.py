from class_data_new import *
plt.rcParams['axes.prop_cycle'] = ("cycler('color', ['r', 'g', 'b', 'y']) ")
plt.rcParams['axes.prop_cycle'] = ("cycler('color', ['r','c','m','y','k','b','g','r','c','m']) ")
#'rcmykbgrcm'
plt.rc('legend',fontsize=15) # using a size in points

Data = Wind_Data('/home/jo/DATA/Fluid_Dynamic/Data_Centerline_FractalGrid_fs60kHz.txt', sampling_frequency=60000., plot_path ='/home/jo/Dropbox/Fluiddynamik/Protokoll/PLOTS/LAB/')

Data.plot_time_series(resampling_time=0.1, filename='10_min_average.png')

Data.plot_pdf_all(filename = 'PDF_ALL.png')

Data.plot_different_characteristics_multiplot(intervals=[0.001,0.01, 1], filename= 'Characteristics_multiplot.png')

Data.plot_time_dependence(filename= 'time_dependence.png')

Data.plot_time_dependent_pdf(intervals=[0.00002, 0.0002, 0.02],  filename= 'time_dependence_pdf.png')

Data.plot_power_spectrum(filename='Power_Spectrum.png' , const=70000000)

Data.plot_autocorrelation(filename='Autocorrelation.png')

Data.plot_Khinchin(filename='Khinchin.png')

Data.plot_joint_probabilities( tau=0.0001, bins=50, filename='Joint_probabilities_low.png')

Data.plot_joint_probabilities( tau=0.001, bins=50, filename='Joint_probabilities_high.png')

Data.plot_taylor_length(filename='Taylor_length.png')

Data.plot_velocity_increment(filename= 'velocity_increment.png')

Data.plot_structure_function(filename='structure_function_higher.png')

Data.plot_scaling_S3(filename='S3_scaling.png')

Data.plot_scaling_r(l_min=0.002,l_max=0.02,filename='r_scaling.png')

Data.plot_incremental_pdf([1,100,1000,10000], bar=False, filename='Incremental_pdf.png')

Data.plot_incremental_pdf([1,100,1000,10000], bar=False, filename='Incremental_pdf_not.png', normalize=False)


Data.plot_N_point(44 , 88, projection_lag=50 , bins=100)

l0=(Data.taylor_length/Data.dt).astype(int)
Data.plot_N_point(l0 , 2*l0, projection_lag=25 , bins=60, filename='conditional_probs_1.png')