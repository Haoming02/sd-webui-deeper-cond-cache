# SD Webui Deeper Cond Cache
This is an Extension for the [Automatic1111 Webui](https://github.com/AUTOMATIC1111/stable-diffusion-webui), which caches the `Conditioning` locally on disk for Optimization.

> Apparently in **Automatic1111 Webui** *(but not **Forge** for whatever reason)*, you need to add `--disable-safe-unpickle` to the `COMMANDLINE_ARGS` for this to work

**Important:** This Extension is experimental. Use it at your own risk!

## Introduction
When you took a look at the **Optimizations** section of the **Settings** tab, 
you would see a `Persistent cond cache` option. This caches the conditioning from the previous generation,
so that when you're just changing the seeds, it doesn't have to calculate the conditioning again every single time.

Why not take it a step further? Instead of only caching **one** *(afaik)* conditioning, store them locally on disk instead.
So now even when you boot up the Webui the next day, it will not need to calculate the conditionings again *(provided that you use the same parameters of course)*.

## Implementation
- The database is based on [diskcache](https://pypi.org/project/diskcache/)
- `Cond` and `Uncond` use separated databased
- The maximum size is set to `64 MB` 
    - Each SDXL conditioning is around `600 KB` in my experience
- When full, it would delete the entries that were not used for the longest

## Benchmarks
On my system running [Forge](https://github.com/lllyasviel/stable-diffusion-webui-forge), 
it took around `0.5 seconds` to switch to CLIP and calculate the conditioning for SDXL; 
meanwhile, it took less than `0.01 seconds` to calculate the key to retrive the conditioning from cache.

This only benefits with regular usages of the exact same prompts. If you're *creative*, you would **not** need this.
