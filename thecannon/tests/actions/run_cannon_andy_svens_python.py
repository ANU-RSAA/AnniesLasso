#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import thecannon as tc


# In[ ]:


# Read in the training labels and model spectra
# These are 1000 samples of labels 'EWT','LMA','N','CHL','CAR','ANT'
# used to modelled spectra at wavelengths 400..1..2500nm with the PROSPECT code
LUC_labels  = pd.read_csv('tests/data/PROSPECT_input.csv')
LUC_spectra = pd.read_csv('tests/data/PROSPECT_LUT.csv')


# In[ ]:


# Prepare input for a complete and restrictred Cannon model
prospect_wavelength = np.arange(400,2501,1)

labels = ['EWT','LMA','N','CHL','CAR','ANT']
prospect_labels = np.array(LUC_labels[labels])

labels_restricted = ['EWT','LMA']
prospect_labels_restricted = np.array(LUC_labels[labels_restricted])

prospect_spectra = np.array(LUC_spectra).T
prospect_spectra_ivar = (100./prospect_spectra)**2.0 # SNR 100


# In[ ]:


# Initialise and train the complete Cannon Model
prospect_model = tc.CannonModel(
    prospect_labels,
    prospect_spectra, prospect_spectra_ivar,
    vectorizer=tc.vectorizer.PolynomialVectorizer(list(labels), 2),dispersion=prospect_wavelength)
prospect_theta, prospect_s2, prospect_metadata = prospect_model.train(threads=1)


# In[ ]:


# Test the label recovery of the same spectra
prospect_test_labels, prospect_test_cov, prospect_metadata = prospect_model.test(prospect_spectra, prospect_spectra_ivar)


# In[ ]:


prospect_model.write('prospect_model_svens_python.model',overwrite=True)


# In[ ]:


# Initialise and train the restricted Cannon Model
prospect_model_restricted = tc.CannonModel(
    prospect_labels_restricted,
    prospect_spectra, prospect_spectra_ivar,
    vectorizer=tc.vectorizer.PolynomialVectorizer(list(labels_restricted), 2),dispersion=prospect_wavelength)
prospect_theta_restricted, prospect_s2_restricted, prospect_metadata_restricted = prospect_model_restricted.train(threads=1)


# In[ ]:


# Test the label recovery of the same spectra
prospect_test_labels_restricted, prospect_test_cov_restricted, prospect_metadata_restricted = prospect_model_restricted.test(prospect_spectra, prospect_spectra_ivar)


# In[ ]:


prospect_model_restricted.write('prospect_model_restricted_test.model',overwrite=True)

