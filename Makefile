env:
	devbox shell

download_data:
	dvc pull

update-env:
	poetry update

push_images:
	@rm -r /home/gormat/Documents/MGR-Praca-Dyplomowa/reports || true 2&> /dev/null
	@cp -r ./reports ../MGR-Praca-Dyplomowa/
	cd ../MGR-Praca-Dyplomowa && git add ./reports && git commit -m "Update reports" && git push origin
