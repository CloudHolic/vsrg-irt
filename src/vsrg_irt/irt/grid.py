from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Iterable, Iterator, Optional

from ..specs import DataSpec, FitConfig


# Enumeration ------------------------------------------------------------------

def cell_key(cfg: FitConfig) -> tuple[str, str, str]:
    return cfg.model, cfg.spec.response, cfg.spec.sample


def _label(cfg: FitConfig) -> str:
    return f"{cfg.spec.key}K {cfg.model}/{cfg.spec.response}/{cfg.spec.sample}"


def grid_configs(key: int=4, *, sample_sets: Iterable[str] = ("random", "all"),
                 models: Optional[Iterable[str]]=None,
                 min_item: int=2, min_user: int=2, **fit_overrides) -> Iterator[FitConfig]:
    from .registry import valid_combos

    combos = sorted(valid_combos())
    if models is not None:
        keep = set(models)
        combos = [(m, r) for (m, r) in combos if m in keep]

    for model, response in combos:
        for sample in sample_sets:
            spec = DataSpec(key=key, response=response, sample=sample, min_item=min_item, min_user=min_user)
            yield FitConfig(spec=spec, model=model, **fit_overrides)


# One cell ---------------------------------------------------------------------

def run_cell(cfg: FitConfig, *, with_persons: bool=True, progress: bool=False) -> Optional[dict]:
    from ..data import load_dataset
    from .registry import get_model
    from . import inference

    try:
        dataset = load_dataset(cfg.spec, allow_db=False)
    except FileNotFoundError as e:
        print(f"[{_label(cfg)}] no cached data - SKIP ({e})", flush=True)
        return None

    model = get_model(cfg)
    data = model.make_data(dataset)
    print(f"[{_label(cfg)}] persons={dataset.n_persons} items={dataset.n_items} "
          f"obs={dataset.n_obs}", flush=True)

    guide, res = inference.fit(model, data, cfg, progress=progress)
    out = {"cfg": cfg, "items": inference.item_intervals(model, guide, res.params, dataset, data, cfg)}
    if with_persons:
        out["persons"] = inference.person_intervals(model, guide, res.params, dataset, cfg)
    return out


# Whole grid -------------------------------------------------------------------

def warm_cache(specs):
    from ..data import load_dataset
    ok = set()
    for spec in specs:
        try:
            load_dataset(spec, allow_db=True)
            ok.add(spec)
        except ValueError as e:
            print(f"[{spec.cache_name}] empty - SKIP ({e})", flush=True)

    return ok


def run_grid(key=4, *, jobs=1, with_persons=True, threads_per_worker=1, **cfg_kw):
    cfgs = list(grid_configs(key=key, **cfg_kw))
    ok = warm_cache({c.spec for c in cfgs})
    cfgs = [c for c in cfgs if c.spec in ok]

    results = {}
    if jobs <= 1:
        for c in cfgs:
            r = run_cell(c, with_persons=with_persons, progress=True)
            if r is not None:
                results[cell_key(c)] = r

        return results

    with ProcessPoolExecutor(max_workers=jobs, initializer=configure_cpu, initargs=(threads_per_worker,)) as ex:
        futs = {ex.submit(run_ceil, c, with_persons=with_persons, progress=False): c
                for c in cfgs}

        for fut in as_completed(futs):
            c = futs[fut]
            try:
                r = fut.result()
                if r is not None:
                    results[cell_key(c)] = r
            except Exception as e:
                print(f"[{_label(c)}] FAILED: {e}", flush=True)

    return results
