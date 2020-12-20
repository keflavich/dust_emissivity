Dust Emissivity & Mass Calculation Tools
========================================

Simple tools built on astropy's units & constants framework for computing the
mass / column of a given chunk of dust.

Compare to https://code.google.com/p/agpy/source/browse/trunk/agpy/blackbody.py
and https://code.google.com/p/agpy/source/browse/trunk/agpy/dust.py, which are
standalone and do not make use of the units framework.

Installation
------------

Installation is through normal pip:

.. code-block:: bash

   pip install https://github.com/keflavich/dust_emissivity/archive/master.zip


Then, one can import:

.. code-block:: python

   import dust_emissivity
   from dust_emissivity import blackbody, modified_blackbody, integrate_sed

Documentation
-------------

There is an `example notebook <http://keflavich.github.io/dust_emissivity/example/Luminosity.html>`_ and docstrings
