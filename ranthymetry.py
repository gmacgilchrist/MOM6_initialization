"""

Tom Bolton
26/03/2019
thomasmichaelbolton@gmail.com

Given a 2D spatial field, generate "random" samples
of the field with the same 2D spatial power spectrum.


"""

import numpy as np

def gen_samples( f, n_samples, apply_hamming=True ) :
    """
    Estimate the 2D power spectrum of the input
    field f, and use random phase sampling to produce
    n_samples of f with the same spectrum of f.
    
    Inputs
    f : 2D array, field to produce samples of
    n_samples : integar, number of samples to generate
    
    Outputs
    samples : 3D array containing the samples of 2D fields
    
    """
    Ny, Nx = f.shape
    
    # applying Hamming window
    if apply_hamming :
        
        w_x = np.hamming(Nx)
        w_y = np.hamming(Ny)
        
        win = np.sqrt( np.outer( w_y, w_x ) )
        
        f *= win
    
    samples = np.zeros( (n_samples,Ny,Nx) )
    samples_spec = np.empty( (n_samples,Ny,Nx) )
    
    # calculate mean and std of f
    f_mean = np.mean(f)
    f_sig = np.std(f)
    
    # remove mean
    f -= f_mean
    
    # 2D power spectrum calculation
    dft = np.fft.fft2(f)            # compute 2D dft
    #dft = np.fft.fftshift( psd )    # shift zero frequency
    
    mod = np.abs(dft)
    
    for i in range(n_samples) :
        
        # generate 2D grid of random phases
        # between 0 and 2pi
        phase_x = np.zeros( Nx )
        phase_y = np.zeros( Ny )
        # In the following, impose zero in first entry of randomised phases
        # Procedure:
            # Randomise entries in phase_* from index 1 to Nx/2
            # (remember in python, index Nx/2 is the entry one greater than the mid-point)
                # e.g. phase_x = [0 1 2 3 4 0 0 0]
            # Take entries 1 to Nx/2, set them to negative, reverse their order,
            # and put them into phase_* in entries Nx/2 to end
                # e.g. phase_x = [0 1 2 3 -4 -3 -2 -1]
            # Note that index Nx/2 is set in the first step and then just reversed in sign
            # For an odd number of entries, the entry in index Nx/2 is set twice...
                # e.g. step 1: phase_x = [0 1 2 3 4 0 0 0 0]
                # then step 2: phase_x = [0 1 2 3 4 -4 -3 -2 -1]
        
        if Nx % 2 == 0 :
            
            phase_x[1:int(Nx/2+1)] = np.random.uniform( size=int(Nx/2) ) * 2 *np.pi
            phase_x[int(Nx/2):] = np.flip(-phase_x[1:int(Nx/2+1)])
            
        else :
            
            phase_x[1:int((Nx-1)/2+1)] = np.random.uniform( size=int( (Nx-1)/2) ) * 2 *np.pi
            phase_x[int((Nx-1)/2+1):] = np.flip(-phase_x[1:int((Nx-1)/2+1)])
            
        if Ny % 2 == 0 :
            
            phase_y[1:int(Ny/2+1)] = np.random.uniform( size=int(Ny/2) ) * 2 *np.pi
            phase_y[int(Ny/2):] = np.flip(-phase_y[1:int(Ny/2+1)])
            
        else :
            
            phase_y[1:int((Ny-1)/2+1)] = np.random.uniform( size=int( (Ny-1)/2) ) * 2 *np.pi
            phase_y[int((Ny-1)/2+1):] = np.flip(-phase_y[1:int((Ny-1)/2+1)])
            
        
        phase_x, phase_y = np.meshgrid( phase_x, phase_y )
            
        # use random phases to construct
        # new 2D sample with same spectrum
        f_spec = mod * np.exp( 1j * phase_x + 1j * phase_y  )
        
        # invert the sample power spectrum
        # and reintroduce mean
        f_sample = np.real( np.fft.ifft2( f_spec ) )
        f_sample += f_mean
        
        # re-scale to variance of original f
        f_sample /= np.std( f_sample )
        f_sample *= f_sig
        
        samples[i,:,:] = f_sample
        samples_spec[i,:,:] = f_spec
        
        
    return samples, samples_spec, dft
        
        
        
    
    
    
    