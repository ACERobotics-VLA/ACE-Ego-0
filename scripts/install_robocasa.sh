#!/usr/bin/env bash
set -euo pipefail

# These revisions are the simulator stack used for the released evaluation.
ROBO_SUITE_REPOSITORY="https://github.com/ARISE-Initiative/robosuite"
ROBO_SUITE_REVISION="51cc01785bab80ffeed20da15e67d7dd4140e76a"
ROBOCASA_REPOSITORY="https://github.com/robocasa/robocasa-gr1-tabletop-tasks"
ROBOCASA_REVISION="4840e671596f93ca03651524b9f72ffb1aadfeff"

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
third_party_root="${ROBOCASA_THIRD_PARTY_DIR:-${repo_root}/third_party}"
python_bin="${PYTHON_BIN:-python}"
robosuite_dir="${third_party_root}/robosuite"
robocasa_dir="${third_party_root}/robocasa-gr1-tabletop-tasks"

die() {
    printf 'error: %s\n' "$1" >&2
    exit 1
}

command -v "${python_bin}" >/dev/null 2>&1 || die "Python executable not found: ${python_bin}"

install_git_checkout() {
    local repository="$1"
    local revision="$2"
    local destination="$3"

    if [[ -n "$(git -C "${destination}" status --porcelain)" ]]; then
        die "Refusing to change ${destination}: the checkout has local modifications."
    fi

    git -C "${destination}" fetch --depth 1 origin "${revision}"
    git -C "${destination}" checkout --detach "${revision}"
}

install_source_archive() {
    local repository="$1"
    local revision="$2"
    local destination="$3"
    local staging_dir
    local archive_path
    local archive_url

    command -v tar >/dev/null 2>&1 || die "tar is required to extract simulator sources."
    if command -v curl >/dev/null 2>&1; then
        download_tool=(curl --fail --location --retry 3 --retry-delay 2 --connect-timeout 30 --silent --show-error)
    elif command -v wget >/dev/null 2>&1; then
        download_tool=(wget --tries=3 --timeout=30)
    else
        die "curl or wget is required to download simulator sources."
    fi

    if [[ -e "${destination}" ]]; then
        if [[ -f "${destination}/.ace_ego_release_revision" ]]; then
            local installed_revision
            installed_revision="$(<"${destination}/.ace_ego_release_revision")"
            [[ "${installed_revision}" == "${revision}" ]] || die "${destination} is pinned to ${installed_revision}, expected ${revision}."
            return
        fi
        die "Refusing to use ${destination}: it is not a pinned source archive or Git checkout."
    fi

    staging_dir="$(mktemp -d "$(dirname -- "${destination}")/.staging.XXXXXX")"
    archive_path="${staging_dir}/source.tar.gz"
    archive_url="${repository}/archive/${revision}.tar.gz"
    if [[ "${download_tool[0]}" == "curl" ]]; then
        "${download_tool[@]}" --output "${archive_path}" "${archive_url}"
    else
        "${download_tool[@]}" --output-document "${archive_path}" "${archive_url}"
    fi
    mkdir "${staging_dir}/source"
    tar -xzf "${archive_path}" -C "${staging_dir}/source" --strip-components=1
    printf '%s\n' "${revision}" >"${staging_dir}/source/.ace_ego_release_revision"
    mv "${staging_dir}/source" "${destination}"
    rmdir "${staging_dir}"
}

install_repository() {
    local repository="$1"
    local revision="$2"
    local destination="$3"

    mkdir -p "$(dirname -- "${destination}")"
    if [[ -d "${destination}/.git" ]]; then
        command -v git >/dev/null 2>&1 || die "git is required to update an existing checkout."
        install_git_checkout "${repository}.git" "${revision}" "${destination}"
    else
        install_source_archive "${repository}" "${revision}" "${destination}"
    fi
}

install_repository "${ROBO_SUITE_REPOSITORY}" "${ROBO_SUITE_REVISION}" "${robosuite_dir}"
install_repository "${ROBOCASA_REPOSITORY}" "${ROBOCASA_REVISION}" "${robocasa_dir}"

"${python_bin}" -m pip install -e "${robosuite_dir}"
"${python_bin}" -m pip install -e "${robocasa_dir}"

if [[ "${SKIP_ROBOCASA_ASSETS:-0}" == "1" ]]; then
    printf 'Skipping RoboCasa tabletop asset download because SKIP_ROBOCASA_ASSETS=1.\n'
    exit 0
else
    "${python_bin}" "${robocasa_dir}/robocasa/scripts/download_tabletop_assets.py" --yes
fi

"${python_bin}" "${repo_root}/scripts/check_environment.py"
