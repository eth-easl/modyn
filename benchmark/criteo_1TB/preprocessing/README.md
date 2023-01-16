# Criteo 1TB Dataset Preprocessing for Modyn

The preprocessing for the Criteo 1TB dataset is based on the steps provided in the [NVIDIA repository](https://github.com/NVIDIA/DeepLearningExamples/blob/master/PyTorch/Recommendation/DLRM/README.md). However some changes were required to process the data in the format requried for the benchmark tests done for Modyn.


## Detailed steps
Here is a set of detailed steps for setting up a google cloud console and pre processing the data from scratch including changes to the script files provided in the NVIDIA repository.


Setup of machines:
1. Create Instance - I used an n1-standard-8 VM with a V100 GPU attached.
2. Create and attach Disk on Cloud Dashboard - Use at least a 4TB hardisk as the Spark Temporary output will cause the size to go above 2TB at least.
3. SSH into instance (gcloud compute ssh <vm_name> after doing gcloud init)
4. Perform first time installations:
	1. Find the name of the attached disk: `sudo lsblk`
    2. Mount external hard disk to a fixed location:  
	`sudo mount -o discard,defaults /dev/sdb /mnt/disks/criteo`  
	`sudo chmod a+w /mnt/disks/criteo`
	3. Install wget: `sudo apt-get install wget`
	4. Install docker:  
		i. `mkdir docker`  
		ii. `cd docker`  
		iii. wget https://download.docker.com/linux/debian/dists/bullseye/pool/stable/amd64/containerd.io_1.6.9-1_amd64.deb  
		iv. wget https://download.docker.com/linux/debian/dists/bullseye/pool/stable/amd64/docker-ce-cli_20.10.9~3-0~debian-bullseye_amd64.deb  
		v. wget https://download.docker.com/linux/debian/dists/bullseye/pool/stable/amd64/docker-ce_20.10.9~3-0~debian-bullseye_amd64.deb  
		vi. wget https://download.docker.com/linux/debian/dists/bullseye/pool/stable/amd64/docker-compose-plugin_2.6.0~debian-bullseye_amd64.deb  
		vii. Install the docker packages using `dpkg -I ./<file>.deb`  
		viii. Test -  
		`sudo service docker start`  
		`sudo docker run hello-world`  

	5. Install git: sudo apt-get install git


Preprocessing Data:
1. Download the criteo dataset on the shared disk -  
`cd /mnt/disks/criteo/data`  
``curl -O https://sacriteopcail01.z16.web.core.windows.net/day_{`seq -s “,” 0 23`}.gz``
2. Clone the dlrm repo: `git clone https://github.com/NVIDIA/DeepLearningExamples`
3. Set the mount location - `vi .bashrc`  
CRITEO_DATASET_PARENT_DIRECTORY=/mnt/disks/criteo/data  
`source .bashrc`
4. Got to the repository directory: `cd DeepLearningExamples/PyTorch/Recommendation/DLRM`
5. Build docker image: `sudo docker build -t nvidia_dlrm_preprocessing -f Dockerfile_preprocessing . --build-arg DGX_VERSION=DGX-A100`
6. Setup if using GPU: 
	1. [cuda - Add nvidia runtime to docker runtimes - Stack Overflow](https://stackoverflow.com/questions/59008295/add-nvidia-runtime-to-docker-runtimes)
	2. [Migration Notice | nvidia-docker](https://nvidia.github.io/nvidia-docker/)
	3. [GitHub - NVIDIA/nvidia-container-runtime: NVIDIA container runtime](https://github.com/nvidia/nvidia-container-runtime#daemon-configuration-file)
	4. Get Nvidia driver. For that first install gcc: `sudo apt install build-essential`
	5. Download Nvidia driver file: `wget https://us.download.nvidia.com/XFree86/Linux-x86_64/525.60.11/NVIDIA-Linux-x86_64-525.60.11.run`  
	`chmod 755 NVIDIA-Linux-x86_64-525.60.11.run`
	6. Install linux kernel headers: `sudo apt install linux-headers-$(uname -r)`
	7. Install Nvidia Driver: `sudo sh NVIDIA-Linux-x86_64-$DRIVER_VERSION.run`
	8. Test if it works - `nvidia-smi`
	9. Restart docker: `sudo systemctl daemon-reload ; sudo systemctl restart docker`
7. Login to container - `sudo docker run -it --rm --ipc=host --runtime=nvidia -v ${CRITEO_DATASET_PARENT_DIRECTORY}:/data/dlrm nvidia_dlrm_preprocessing bash` 
8. Set the number of GPUs correctly (0 or 1) in /opt/spark/conf/spark-defaults.conf
9. `cd /workspace/dlrm/preproc`
10. `chmod 755 *`
11. Replace the following edited files in the edited_preproc_scripts folder into the container:  
    a. DGX-A100_config.sh  
    b. run_spark_gpu_DGX-A100.sh  
    c. prepare_dataset.sh  
12. Start the training - `./prepare_dataset.sh 15 GPU Spark`

Uploading Data to GCS:
1. Create a bucket on the Google Console (I've created a bucket called dds-criteo)
2. Cloud init (cause it requires permission)
3. Cd to the mounted directory - `cd /mnt/disks/criteo/`
4. Upload to cloud storage - `gcloud storage cp -r output gs://dds-criteo`