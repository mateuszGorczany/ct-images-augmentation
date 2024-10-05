MODE ?= train
WANDB_RUN_ID ?= ""
MODEL_CHECKPOINT ?= ""
RETRAIN_LEARNING_RATE ?= ""
RETRAIN_EPOCHS ?= ""

env:
	devbox shell

lint:
	pre-commit

download_data:
	dvc pull

push_images:
	@rm -r /home/gormat/Documents/MGR-Praca-Dyplomowa/reports || true 2&> /dev/null
	@cp -r ./reports ../MGR-Praca-Dyplomowa/
	cd ../MGR-Praca-Dyplomowa && git pull && git add ./reports && git commit -m "Update reports" && git push origin

clean:
	rm nohup.out || true 2&> /dev/null
	-pkill -f "python ./src/main.py"
	-pkill -f "/home/gormat/Documents/ct-images-augmentation/.venv/bin/python /home/gormat/Documents/ct-images-augmentation/src/main.py"  || true
	-pkill wandb-service || true

# Define a template command
run_experiment = nohup devbox run python \
		./src/main.py \
		--mode $(MODE) \
		--wandb-run-id $(WANDB_RUN_ID) \
		--model-checkpoint $(MODEL_CHECKPOINT) \
		--retrain-learning-rate $(RETRAIN_LEARNING_RATE) \
		--retrain-epochs $(RETRAIN_EPOCHS)

# Define individual targets using the template command
experiment-wgan: clean
	$(run_experiment) --config=./experiments/wgan.yaml

experiment-monai_autoencoder: clean
	$(run_experiment) --config=./experiments/monai_autoencoder.yaml

experiment-monai_diffuser: clean
	$(run_experiment) --config=./experiments/monai_diffuser.yaml

experiment-meta_diffuser: clean
	$(run_experiment) --config=./experiments/meta_diffuser.yaml

experiment-meta_vqgan: clean
	$(run_experiment) --config=./experiments/meta_vqgan.yaml

experiment-vision_gan: clean
	$(run_experiment) --config=./experiments/gan.yaml

experiment-mini_gan: clean
	$(run_experiment) --config=./experiments/mini_gan.yaml

experiment-german_vqvae: clean
	nohup devbox run python ./src/models/transformers_ct_reconstruction/vqgan/train.py

experiment-german_vqgan: clean
	nohup devbox run python \
		./src/models/transformers_ct_reconstruction/vqgan/train.py \
		--best-vq-vae-ckpt "./models/german_vqvae/checkpoints/epoch=5149-step=15450.ckpt"

experiment-german_vqvae: clean
	nohup devbox run python \
		./src/models/transformers_ct_reconstruction/vqgan/train.py