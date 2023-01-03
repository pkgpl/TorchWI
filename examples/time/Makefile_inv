NP=4
HOST=localhost:4

DEVICE=cuda

DATADIR=../data_marm
FRDDIR=../forward

# fdm order
ORDER=2

VMIN=1.5
VMAX=5.5

LR=0.01
OPTIM=adam
MAX_EPOCHS=500

inv_cuda:
	horovodrun -np $(NP) -H $(HOST) python inv_torch.py \
		--name "Conventional FWI" \
		--device cuda \
		--fvel $(DATADIR)/vinit.rsf \
		--fwav $(DATADIR)/wavelet.rsf \
		--fshot $(DATADIR)/shot.rsf \
		--ftrue $(FRDDIR)/seismo. \
		--order $(ORDER) \
		--vmin $(VMIN) \
		--vmax $(VMAX) \
		--lr $(LR) \
		--optimizer $(OPTIM) \
		--skip_output 10 \
		--max_epochs $(MAX_EPOCHS)

inv_cpu:
	OMP_NUM_THREADS=$(OMP_NUM_THREADS) horovodrun -np $(NP) -H $(HOST) --mpi-args "--prefix $(CONDA_ENV_ROOT)" \
		python inv_torch.py \
		--name "Conventional FWI" \
		--device cpu \
		--fvel $(DATADIR)/vinit.rsf \
		--fwav $(DATADIR)/wavelet.rsf \
		--fshot $(DATADIR)/shot.rsf \
		--ftrue $(FRDDIR)/seismo. \
		--order $(ORDER) \
		--vmin $(VMIN) \
		--vmax $(VMAX) \
		--lr $(LR) \
		--optimizer $(OPTIM) \
		--skip_output 10 \
		--max_epochs $(MAX_EPOCHS)

