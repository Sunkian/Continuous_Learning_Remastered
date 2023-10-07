[//]: # (cd webapp-streamlit-upgrade)

[//]: # (./run_app.sh)






## Continuous Learning at the Edge Software App

#### How to use
```chmod +x run_app.sh``` 

```./run_app.sh```


#### Mongo DB Database
1. Create at your root /data/db
```shell
mkdir -p /data/db
```
2. After having downloaded MongoDB from the website, add mongod and mongos to your path
```shell
export PATH="/Users/apagnoux/Downloads/mongosh-1.10.6-darwin-arm64/bin:$PATH"
```
3. Run mongod to your /data/db location (starts MongoDB server)
```shell
mongod --dbpath /Users/apagnoux/data/db
```
You can download Mingo.io for further GUI



lsof -i :5003
kill $(lsof -t -i :5003)

1) /Users/apagnoux/Downloads/mongodb-macos-aarch64-7.0.0/bin/mongod --dbpath /Users/apagnoux/data/db
2) And cd /Users/apagnoux/data/db and run mongosh



### Docker 

#### Streamlit 

```sh
docker build -t streamlit_app -f Dockerfile_streamlit .
docker run -d -p 8501:8501 streamlit_app
```
Then go to ```http://localhost:8501```


### Docker compose

```shell
docker-compose build
docker-compose up
docker-compose down
```

Then go to ```http://localhost:8501```

On MongoDB Compass, go to ``mongodb://localhost:27017/``

Start everything :
```shell
./start_dockercompose.sh
```

#### Remove all docker containers/images
1. Remove all containers including their volumes
```shell
docker rm -vf $(docker ps -aq)
```
2. Remove all images
```shell
docker rmi -f $(docker images -aq)
```