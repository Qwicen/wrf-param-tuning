1. Clone or download this repo. `cd` yourself to it's root directory.

2. Build docker image \
`docker build -t wrf-params .`

3. Run container, mount volume with geographical static data \
`docker run -it -v /local/path/to/data/:/data wrf-params`
4. Set **LAT**, **LON** parameters at config file \
`sudo vim pipeline/config/config.yml`

5. Run optimization \
`sudo bash pipeline/run.sh`