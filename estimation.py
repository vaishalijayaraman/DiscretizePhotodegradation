
# PARAMETERS

I0 = 0 #intensity of incident light (mWcm-2)
I* = 0 #intensity of light (mWcm-2)

lambda = 365 #(nm)

alpha_pd = 8061 #absorptivity of undegraded gel (M-1cm-1)
alpha_deg = 6073 #absorptivity of degraded gel (M-1cm-1)

c_pd #concentration of undegraded gel
c_deg #concentration of degraded gel

k = 3.3 * (10**(-4)) #kinetic degradation constant (cm2mW-1s-1)

# Function that calculates the concentration of the undegraded sample
def computeConcentration:
  value = I0 * c_pd
  return -1 * value
  
