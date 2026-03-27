
all: docker/files/aws/credentials.enc

docker/files/aws/credentials.enc: docker/files/aws/credentials
	openssl enc -aes-256-cbc -in docker/files/aws/credentials -out docker/files/aws/credentials.enc -K $$mykey -iv $$iv

build: docker/files/aws/credentials.enc
	cd docker ; docker build -t fenyoa/ft_gpt_oss_20b_ovh_faq -f Dockerfile .

run:
	docker run --gpus all --user=42420:42420 -e iv=$$iv -e mykey=$$mykey --rm -t -i fenyoa/ft_gpt_oss_20b_ovh_faq

shell:
	docker run --gpus all --user=42420:42420 -e iv=$$iv -e mykey=$$mykey --rm -t -i fenyoa/ft_gpt_oss_20b_ovh_faq bash

run-on-ovh:
	echo RUN '"ovhai login" (cf. 1Password)'
	@echo ovhai job run --name faq --flavor h100-1-gpu --gpu 1 --ssh-public-keys \"$$sshkey\" --unsecure-http fenyoa/ft_gpt_oss_20b_ovh_faq -e mykey=$$mykey -e iv=$$iv -e wandbkey=$$wandbkey -e hfkey=$$hfkey

push:
	echo RUN '"docker login -u fenyoa"' and enter password
	docker push fenyoa/ft_gpt_oss_20b_ovh_faq

pull:
	rm -rf /mnt/e/gpt-oss-20b-merged-mxfp4
	aws s3 cp s3://cnam-models/gpt-oss-20b-merged-mxfp4 /mnt/e/gpt-oss-20b-merged-mxfp4 --recursive
