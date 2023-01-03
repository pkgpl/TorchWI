NP=4
HOST=localhost:4
# omp for cpu only
OMP_NUM_THREADS=2

DATADIR=../data_marm
FRDDIR=../forward

# fdm order
ORDER=8


forward_cuda:
	horovodrun -np $(NP) -H $(HOST) python forward_modeling.py \
		--device cuda \
		--fvel $(DATADIR)/marm.rsf \
		--fwav $(DATADIR)/wavelet.rsf \
		--fshot $(DATADIR)/shot.rsf \
		--order $(ORDER) \
		--outdir $(FRDDIR)

forward_cpu:
	OMP_NUM_THREADS=$(OMP_NUM_THREADS) horovodrun -np $(NP) -H $(HOST) --mpi-args "--prefix $(CONDA_ENV_ROOT)" \
		python forward_modeling.py \
		--device cpu \
		--fvel $(DATADIR)/marm.rsf \
		--fwav $(DATADIR)/wavelet.rsf \
		--fshot $(DATADIR)/shot.rsf \
		--order $(ORDER) \
		--outdir $(FRDDIR)
