#!/bin/bash
# Shared fresh-run configuration for the SeaHub production SGE chain.

seahub_configure_run() {
    local results_base="$1"
    local derived_base="$2"

    if [[ -z "${SEAHUB_RUN_ID:-}" ]]; then
        echo "ERROR: SEAHUB_RUN_ID is required (for example 20260731_prod01)." >&2
        return 2
    fi
    if [[ ! "${SEAHUB_RUN_ID}" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
        echo "ERROR: invalid SEAHUB_RUN_ID=${SEAHUB_RUN_ID@Q}." >&2
        return 2
    fi

    if [[ -z "${SEAHUB_OPERATIONAL_DATE:-}" ]]; then
        if [[ "${SEAHUB_RUN_ID}" =~ ^([0-9]{8}) ]]; then
            SEAHUB_OPERATIONAL_DATE="${BASH_REMATCH[1]}"
        else
            echo "ERROR: set SEAHUB_OPERATIONAL_DATE=YYYYMMDD when the run id does not begin with a date." >&2
            return 2
        fi
    fi
    if [[ ! "${SEAHUB_OPERATIONAL_DATE}" =~ ^[0-9]{8}$ ]]; then
        echo "ERROR: SEAHUB_OPERATIONAL_DATE must be YYYYMMDD." >&2
        return 2
    fi

    SEAHUB_RUN_ROOT="${SEAHUB_RUN_ROOT:-${results_base}/outputs/full_corpus_${SEAHUB_RUN_ID}}"
    SEAHUB_BUNDLE_ROOT="${SEAHUB_BUNDLE_ROOT:-${derived_base}/${SEAHUB_RUN_ID}/bundle}"
    if [[ "${SEAHUB_RUN_ROOT}" != /* || "${SEAHUB_BUNDLE_ROOT}" != /* ]]; then
        echo "ERROR: SeaHub run and bundle roots must be absolute paths." >&2
        return 2
    fi

    export SEAHUB_RUN_ID SEAHUB_OPERATIONAL_DATE SEAHUB_RUN_ROOT SEAHUB_BUNDLE_ROOT
    SEAHUB_QSUB_ENV="SEAHUB_RUN_ID=${SEAHUB_RUN_ID},SEAHUB_OPERATIONAL_DATE=${SEAHUB_OPERATIONAL_DATE},SEAHUB_RUN_ROOT=${SEAHUB_RUN_ROOT},SEAHUB_BUNDLE_ROOT=${SEAHUB_BUNDLE_ROOT}"
    export SEAHUB_QSUB_ENV
}

