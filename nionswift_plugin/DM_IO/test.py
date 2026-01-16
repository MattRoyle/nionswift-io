import h5py
from DMDelegates import *

delegate = DM5IODelegate(None)

dm = delegate.read_data_and_metadata("dm5", r"C:\Users\Matthew.Royle\Downloads\Test DM5 files\EELS_Spectrum_Image_NiCl2_2.dm5")
delegate.write_data_and_metadata(dm, r"C:\Users\Matthew.Royle\Downloads\Test DM5 files\nice.dm5", "dm5")