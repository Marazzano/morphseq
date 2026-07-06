"""Resolved image-product plan JSONs for product-grain materialization.

These JSON files are execution commitments for one ``(well_id, image_product_key)``. They sit
between config/request resolution and the scope backend, so a product materialization job consumes
exactly one concrete product rather than interpreting the whole run config.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from data_pipeline.acquisition.image_materialization.image_product_keys import build_image_product_key
from data_pipeline.acquisition.image_materialization.materialization_plan import (
    ResolvedImageProduct,
    load_image_materialization_plan,
)
from data_pipeline.acquisition.image_materialization.scope.scope_resolver_for_materialization_plan import (
    resolve_materialization_plan,
)

RESOLVED_PRODUCT_PLAN_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class ResolvedProductPlanForWell:
    """One resolved product commitment for one well."""

    experiment_id: str
    well_id: str
    scope_name: str
    product_key: str
    product: ResolvedImageProduct


def image_product_key_for_resolved_product(product: ResolvedImageProduct) -> str:
    """Return the image product key for one resolved product."""
    return build_image_product_key(
        channel_id=product.channel_id,
        image_product_type=product.image_product_type,
        projection_method=product.projection_method,
    )


def resolve_requested_image_products_for_well(
    *,
    experiment_id: str,
    well_id: str,
    scope_name: str,
    config: dict | None,
) -> tuple[ResolvedProductPlanForWell, ...]:
    """Resolve run config into one product commitment per requested image product."""
    scope_name = str(scope_name).strip().lower()
    requested_plan = load_image_materialization_plan(config)
    resolved_plan = resolve_materialization_plan(
        scope_name=scope_name,
        requested_plan=requested_plan,
    )
    plans = []
    seen: set[str] = set()
    for product in resolved_plan.products:
        product_key = image_product_key_for_resolved_product(product)
        if product_key in seen:
            raise ValueError(
                f"Duplicate resolved image product key {product_key!r} for well {well_id!r}. "
                "Each requested image product must resolve to a distinct product key."
            )
        seen.add(product_key)
        plans.append(
            ResolvedProductPlanForWell(
                experiment_id=str(experiment_id),
                well_id=str(well_id),
                scope_name=scope_name,
                product_key=product_key,
                product=product,
            )
        )
    return tuple(plans)


def write_resolved_product_plan_for_well(
    *,
    experiment_id: str,
    well_id: str,
    scope_name: str,
    config: dict | None,
    product_key: str,
    output_json: Path,
) -> ResolvedProductPlanForWell:
    """Write the selected resolved product plan JSON for one ``(well, product_key)``."""
    plans = resolve_requested_image_products_for_well(
        experiment_id=experiment_id,
        well_id=well_id,
        scope_name=scope_name,
        config=config,
    )
    by_key = {plan.product_key: plan for plan in plans}
    if product_key not in by_key:
        raise ValueError(
            f"Requested product_key {product_key!r} is not active for well {well_id!r}. "
            f"Active product keys: {sorted(by_key)}."
        )

    selected = by_key[product_key]
    output_json = Path(output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(_resolved_product_plan_to_payload(selected), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return selected


def load_resolved_product_plan_for_well(
    path: Path,
    *,
    expected_experiment_id: str | None = None,
    expected_well_id: str | None = None,
    expected_product_key: str | None = None,
) -> ResolvedProductPlanForWell:
    """Load a resolved product plan JSON and optionally assert expected path/wildcard atoms."""
    path = Path(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != RESOLVED_PRODUCT_PLAN_SCHEMA_VERSION:
        raise ValueError(
            f"{path}: unsupported resolved product plan schema_version "
            f"{payload.get('schema_version')!r}; expected {RESOLVED_PRODUCT_PLAN_SCHEMA_VERSION}."
        )

    product_payload = payload["resolved_product"]
    product = ResolvedImageProduct(
        channel_id=product_payload["channel_id"],
        image_product_type=product_payload["image_product_type"],
        projection_method=product_payload.get("projection_method"),
        xy_composition=product_payload["xy_composition"],
    )
    plan = ResolvedProductPlanForWell(
        experiment_id=str(payload["experiment_id"]),
        well_id=str(payload["well_id"]),
        scope_name=str(payload["scope_name"]),
        product_key=str(payload["product_key"]),
        product=product,
    )
    _assert_expected(plan.experiment_id, expected_experiment_id, "experiment_id", path)
    _assert_expected(plan.well_id, expected_well_id, "well_id", path)
    _assert_expected(plan.product_key, expected_product_key, "product_key", path)
    derived_key = image_product_key_for_resolved_product(plan.product)
    if plan.product_key != derived_key:
        raise ValueError(
            f"{path}: product_key {plan.product_key!r} does not match resolved product fields "
            f"(derived {derived_key!r})."
        )
    return plan


def _resolved_product_plan_to_payload(plan: ResolvedProductPlanForWell) -> dict:
    return {
        "schema_version": RESOLVED_PRODUCT_PLAN_SCHEMA_VERSION,
        "experiment_id": plan.experiment_id,
        "well_id": plan.well_id,
        "scope_name": plan.scope_name,
        "product_key": plan.product_key,
        "resolved_product": {
            "channel_id": plan.product.channel_id,
            "image_product_type": plan.product.image_product_type,
            "projection_method": plan.product.projection_method,
            "xy_composition": plan.product.xy_composition,
        },
    }


def _assert_expected(actual: str, expected: str | None, field: str, path: Path) -> None:
    if expected is None:
        return
    if actual != str(expected):
        raise ValueError(
            f"{path}: resolved product plan {field}={actual!r} does not match expected "
            f"{field}={str(expected)!r}."
        )
