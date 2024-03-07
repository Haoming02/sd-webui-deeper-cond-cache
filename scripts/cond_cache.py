from modules.processing import StableDiffusionProcessing as SDP
from modules import shared, scripts, devices, prompt_parser

from hashlib import md5 as to_hash
from json import dumps as to_json
from diskcache import Cache
# import time
import os


cond_cache = Cache(directory=os.path.join(scripts.basedir(), "cond"), size_limit=64e6, cull_limit=16, eviction_policy="least-recently-used")
uncond_cache = Cache(directory=os.path.join(scripts.basedir(), "uncond"), size_limit=64e6, cull_limit=16, eviction_policy="least-recently-used")


def get_conds_with_persistent_caching(self, function, required_prompts, steps, caches, extra_network_data, hires_steps=None):
    """
    Returns the result of calling function(shared.sd_model, required_prompts, steps)
    using a cache to store the result if the same arguments have been used before.

    cond_cache is a diskcache database holding the calculated conds
    uncond_cache is a diskcache database holding the calculated unconds

    caches is a placeholder for compatibility...
    """

    if shared.opts.use_old_scheduling:
        old_schedules = prompt_parser.get_learned_conditioning_prompt_schedules(required_prompts, steps, hires_steps, False)
        new_schedules = prompt_parser.get_learned_conditioning_prompt_schedules(required_prompts, steps, hires_steps, True)
        if old_schedules != new_schedules:
            self.extra_generation_params["Old prompt editing timelines"] = True


    # start_time = time.time()

    cached_params = self.cached_params(required_prompts, steps, extra_network_data, hires_steps, shared.opts.use_old_scheduling)
    params = list(cached_params)

    networks = cached_params[6].get('lora', []) + cached_params[6].get('lyco', []) + cached_params[6].get('hypernet', [])
    network = [to_json(n.items) for n in networks]

    params[0] = cached_params[0][0].replace(" ", "").replace(",", "")
    params[5] = cached_params[5].hash
    params[6] = to_json(network)

    key:str = to_hash(to_json(params).encode('utf-8')).hexdigest()

    # print(f"\n> Calculate Key: {round(time.time() - start_time, 2)} seconds\n")


    if function is prompt_parser.get_learned_conditioning:
        # uncond
        try:
            return uncond_cache[key]

        except KeyError:
            with devices.autocast():
                c = function(shared.sd_model, required_prompts, steps, hires_steps, shared.opts.use_old_scheduling)

            uncond_cache[key] = c
            return c


    elif function is prompt_parser.get_multicond_learned_conditioning:
        # cond
        try:
            return cond_cache[key]

        except KeyError:
            # start_time = time.time()
            with devices.autocast():
                c = function(shared.sd_model, required_prompts, steps, hires_steps, shared.opts.use_old_scheduling)
            # print(f"\n> Calculate Cond: {round(time.time() - start_time, 2)} seconds\n")

            cond_cache[key] = c
            return c


    else:
        raise ValueError("Unrecognized Function...")


SDP.get_conds_with_caching = get_conds_with_persistent_caching
