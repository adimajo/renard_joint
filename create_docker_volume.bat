# Data Harvesting
docker volume create data_data_harvesting
docker container create --name dummy -v data_data_harvesting:/the_data hello-world
docker cp C:\Users\EHRHARA\docker_share\data_harvesting\data dummy:/the_data/
docker cp C:\Users\EHRHARA\docker_share\data_harvesting\model dummy:/the_data/
docker rm dummy
