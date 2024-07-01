env:
	devbox shell

download_data:
	dvc pull

push_images:
	@rm -r /home/gormat/Documents/MGR-Praca-Dyplomowa/reports || true 2&> /dev/null
	@cp -r ./reports ../MGR-Praca-Dyplomowa/
	cd ../MGR-Praca-Dyplomowa && git add ./reports && git commit -m "Update reports" && git push origin


experiment-vision_transformer:
	devbox run python ./src/main.py --config ./experiments/visoin_transformer.yaml

experiment-vision_gan:
	devbox run python ./src/main.py --config ./experiments/gan.yaml
