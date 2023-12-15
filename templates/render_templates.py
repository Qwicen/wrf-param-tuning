import os
import yaml
import argparse
from datetime import datetime
from jinja2 import Environment, FileSystemLoader

def render_wps_config(wrf_root='.'):
    env = Environment(loader=FileSystemLoader(os.path.join(wrf_root, 'pipeline/templates')))
    template_wps = env.get_template('template_namelist.wps')

    with open(os.path.join(wrf_root, 'pipeline/config/config.yml'), "r") as file:
        config = yaml.load(file, Loader=yaml.CLoader)

    time = {key: config['time'][key] for key in config['time']}
    geogrid = {key: config['geogrid'][key] for key in config['geogrid']}
    domains = {key: ",".join([str(dom[key]) for dom in config['domains']]) for key in config['domains'][0].keys()}


    config_wps = template_wps.render({**time, **geogrid, **domains})
    with open(os.path.join(wrf_root, 'WPS', 'namelist.wps'), "w") as file:
        file.write(config_wps)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--wrf_root', type=str)
    parsed_args = parser.parse_args()
    render_wps_config(parsed_args.wrf_root)

