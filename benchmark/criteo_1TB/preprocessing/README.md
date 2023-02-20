# Criteo 1TB Dataset Preprocessing for Modyn

The preprocessing for the Criteo 1TB dataset is based on the steps provided in the [NVIDIA repository](https://github.com/NVIDIA/DeepLearningExamples/blob/master/PyTorch/Recommendation/DLRM/README.md).
However some changes were required to process the data in the format requried for the benchmark tests done for Modyn.
We have created a patch for the respository which contains the changes.
The below steps explain how the data can be regenerated if needed, including the setup of a google cloud machine as well the running of the script with the changes applied.


## Preprocessing Information
The main aim of the preprocessing is to convert the format into something that is easily readable and works will with the system, help reduce the complexity of the data and clean it in the process.
The first is achieved by outputting the data into many binary files that follow a fixed schema. By following a schema we ensure a standard size for each record which allows us to easily access any record in fixed time.
The second is done by re-embedding the categorical features.
First off, we filter out low frequency values, ie, values that occur less than a chosen threshold - T by embedding them all as a default value.
This helps the training as the model learns to ignore that default value rather than learning from multiple infrequent features.
We also re-embedded the values into a range from 1 to the number of unique values (eg 100 if there 100 unique high frequency values in the category), which helps represent the data as int8 or int16 instead of int32, saving space.

This preprocessing is done with the help of Spark. First all files are read to count the frequency of all categorical values and create a dictionary of the counts.
Secondly a spark processes is triggered to apply this transformation to the data in each file which does it in parallel and outputs multiple parquet files.
Finally another script reads the parquet file output by the spark processes and saves it as binary files.

Note: The original NVIDIA script has additional steps which involes merging all the parquet files for the training/validation/test into single large binary files, as well splitting the files by their cateogries (ie one file per category and one file for numerical values).
However since we use a dynamic selection of data, we chose not to use either of the above steps and hence have been removed in the patch.
For more detailed information on the steps performed in the processing, once can check the scripts in the mentioned NVIDIA github repository


## Detailed steps
Here is a set of detailed steps for setting up a google cloud console and pre processing the data from scratch including applying changes to the script files provided in the NVIDIA repository.


Setup of machines:
1. Create Instance - I used an n1-standard-8 VM with a V100 GPU attached.
2. Create and attach Disk on Cloud Dashboard - Use at least a 4TB hardisk as the Spark Temporary output will cause the size to go above 2TB at least.
3. SSH into instance (gcloud compute ssh <vm_name> after doing gcloud init)
4. Perform first time installations:
	1. Find the name of the attached disk:
		```
		sudo lsblk
		```
    2. Mount external hard disk to a fixed location where disk_path is the path for that disk as found in the previous step (eg: /dev/sdb) :
		```
		sudo mount -o discard,defaults ${disk_path} /mnt/disks/criteo
		sudo chmod a+w /mnt/disks/criteo
		```
	3. Install wget:
		```
		sudo apt-get install wget
		```
	4. Install docker:  
		```
		mkdir docker
		cd docker
		wget https://download.docker.com/linux/debian/dists/bullseye/pool/stable/amd64/containerd.io_1.6.9-1_amd64.deb  
		wget https://download.docker.com/linux/debian/dists/bullseye/pool/stable/amd64/docker-ce-cli_20.10.9~3-0~debian-bullseye_amd64.deb  
		wget https://download.docker.com/linux/debian/dists/bullseye/pool/stable/amd64/docker-ce_20.10.9~3-0~debian-bullseye_amd64.deb  
		wget https://download.docker.com/linux/debian/dists/bullseye/pool/stable/amd64/docker-compose-plugin_2.6.0~debian-bullseye_amd64.deb  
		dpkg -I ./containerd.io_1.6.9-1_amd64.deb  ./docker-compose-plugin_2.6.0~debian-bullseye_amd64.deb  ./docker-ce_20.10.9~3-0~debian-bullseye_amd64.deb  ./docker-ce-cli_20.10.9~3-0~debian-bullseye_amd64.deb
		```
		Test the installation -  
		```
		sudo service docker start
		sudo docker run hello-world
		```

	5. Install git:
		```
		sudo apt-get install git
		```

	6. Setup if using GPU: 
		```
		# Setup nvidia runtime (https://github.com/NVIDIA/nvidia-container-runtime#installation)
		sudo apt-get install nvidia-container-runtime
		sudo tee /etc/docker/daemon.json <<EOF
		{
			"runtimes": {
				"nvidia": {
					"path": "/usr/bin/nvidia-container-runtime",
					"runtimeArgs": []
				}
			}
		}
		EOF
		sudo pkill -SIGHUP dockerd

		# Install gcc
		sudo apt install build-essential

		# Download and install Nvidia driver
		cd ~
		mkdir NVIDIA
		cd NVIDIA
		wget https://us.download.nvidia.com/XFree86/Linux-x86_64/525.60.11/NVIDIA-Linux-x86_64-525.60.11.run
		chmod 755 NVIDIA-Linux-x86_64-525.60.11.run
		sudo apt install linux-headers-$(uname -r)
		sudo sh NVIDIA-Linux-x86_64-525.60.11.run

		# Test if it works
		nvidia-smi

		# Restart docker
		sudo systemctl daemon-reload
		sudo systemctl restart docker
		```


Preprocessing Data:
1. Download the criteo dataset on the shared disk -  
	```
	cd /mnt/disks/criteo/data
	curl -O https://sacriteopcail01.z16.web.core.windows.net/day_{`seq -s “,” 0 23`}.gz
	```
2. Set the mount location of the data location in your `.bashrc` file  
	Set CRITEO_DATASET_PARENT_DIRECTORY=/mnt/disks/criteo/data  
	Source the bash file `source ~/.bashrc`
	  
3. Clone the dlrm repo: 
	```
	git clone https://github.com/NVIDIA/DeepLearningExamples
	```
4. Download and apply the patch:
	```
	cd DeepLearningExamples
	curl -O https://github.com/eth-easl/dynamic_datasets_dsl/tree/main/benchmark/criteo_1TB/preprocessing/modyn.patch
	git apply modyn.patch
	```
5. Build docker image:
	```
	cd ~/DeepLearningExamples/PyTorch/Recommendation/DLRM
	sudo docker build -t nvidia_dlrm_preprocessing -f Dockerfile_preprocessing . --build-arg DGX_VERSION=DGX-A100
	```
	Note: If you are not using a GPU, you can still set the version argument as DGX-A100, but then run the preprocessing script with a different argument

6. Login to container - 
	```
	cd ~/DeepLearningExamples/PyTorch/Recommendation/DLRM
	sudo docker run -it --rm --ipc=host --runtime=nvidia -v ${CRITEO_DATASET_PARENT_DIRECTORY}:/data/dlrm nvidia_dlrm_preprocessing bash
	```

7. Set the number of GPUs in /opt/spark/conf/spark-defaults.conf. If you are not using GPUs, then set it to 0
8. Change to the preprocessing directory
	```
	cd /workspace/dlrm/preproc`
	chmod 755 *
	```
9. Start the preprocessing by running the following command
	```
	./prepare_dataset.sh <frequency-threshold> <CPU|GPU> Spark
	```
	The frequency-threshold parameter controls what consists of "high" frequency vs "low" frequency features as explained in the Preprocessing Information section
	The second argument is to specify if you want to use the CPU or GPU processing.
10. Adjust the file timestamps to reflect the individual days using the script we provide.
	For this, identify the folder in which the final processed binary dataset resides (should be `/mnt/disks/criteo/binary_dataset`).
	Then, run
	```
	python criteo_timestamps.py <DIRECTORY_PATH>
	```
	to adjust the modified timestamps such that Modyn recognizes the files as from multiple days.

Uploading Data to GCS:
1. Create a bucket on the Google Console eg "dds-criteo"
2. Initialize gcloud with your credentials
	```
	gcloud compute init
	```
3. Upload the processed data to the cloud storage - 
	```
	cd /mnt/disks/criteo/
	gcloud storage cp -r output gs://dds-criteo
	```