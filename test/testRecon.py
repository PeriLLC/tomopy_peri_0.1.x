import tomopy
import pylab

obj = tomopy.shepp3d() # Generate an object.
ang = tomopy.angles(180) # Generate uniformly spaced tilt angles.
sim = tomopy.project(obj, ang) # Calculate projections.
#rec = tomopy.recon_accelerated(sim, ang, algorithm='ospml_quad', hardware='Xeon_Phi', implementation='tomoperi') # Reconstruct object.

rec = tomopy.recon_accelerated(sim, ang, algorithm='ospml_hybrid')

# Show 64th slice of the reconstructed object.
pylab.imshow(rec[64], cmap='gray')
pylab.show()

