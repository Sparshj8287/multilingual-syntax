#!/bin/bash

# Usage: bash download_ud.sh [lang_code] [optional_treebank_name]
# Example: bash download_ud.sh hi
# Example: bash download_ud.sh en UD_English-EWT

LANG_CODE=$1
TREEBANK_NAME=$2
VERSION="2.17"
BRANCH="r$VERSION"

if [ -z "$LANG_CODE" ]; then
    echo "Usage: bash download_ud.sh [lang_code] [optional_treebank_name]"
    echo "Example: bash download_ud.sh hi"
    exit 1
fi

# Default mappings for common languages
if [ -z "$TREEBANK_NAME" ]; then
    case $LANG_CODE in
        en) TREEBANK_NAME="UD_English-EWT" ;;
        hi) TREEBANK_NAME="UD_Hindi-HDTB" ;;
        fr) TREEBANK_NAME="UD_French-GSD" ;;
        es) TREEBANK_NAME="UD_Spanish-AnCora" ;;
        ar) TREEBANK_NAME="UD_Arabic-PADT" ;;
        zh) TREEBANK_NAME="UD_Chinese-GSD" ;;
        de) TREEBANK_NAME="UD_German-GSD" ;;
        fi) TREEBANK_NAME="UD_Finnish-TDT" ;;
        id) TREEBANK_NAME="UD_Indonesian-GSD" ;;
        lv) TREEBANK_NAME="UD_Latvian-LVTB" ;;
        fa) TREEBANK_NAME="UD_Persian-Seraji" ;;
        *)
            echo "Error: No default treebank for '$LANG_CODE'. Please specify the full UD repository name."
            echo "Example: bash download_ud.sh ja UD_Japanese-GSD"
            exit 1
            ;;
    esac
fi

DATASETS_DIR="datasets"
LANG_DIR="$DATASETS_DIR/$LANG_CODE"

mkdir -p "$LANG_DIR"

echo "Downloading $TREEBANK_NAME (version $VERSION)..."

# Download the ZIP of the specific branch
TEMP_ZIP="temp_ud.zip"
URL="https://github.com/UniversalDependencies/$TREEBANK_NAME/archive/refs/heads/$BRANCH.zip"

curl -L "$URL" -o "$TEMP_ZIP"

if [ $? -ne 0 ]; then
    echo "Error: Failed to download from $URL"
    rm -f "$TEMP_ZIP"
    exit 1
fi

# Unzip into the language directory
unzip -q "$TEMP_ZIP" -d "$LANG_DIR"
rm "$TEMP_ZIP"

# The unzipped folder will be named like UD_Hindi-HDTB-r2.17
UNZIPPED_FOLDER=$(ls "$LANG_DIR" | grep "$TREEBANK_NAME")

# Move .conllu files and rename them
# We look for files ending in -ud-train.conllu, -ud-dev.conllu, -ud-test.conllu
mv "$LANG_DIR/$UNZIPPED_FOLDER"/*-ud-train.conllu "$LANG_DIR/train.conllu" 2>/dev/null
mv "$LANG_DIR/$UNZIPPED_FOLDER"/*-ud-dev.conllu "$LANG_DIR/dev.conllu" 2>/dev/null
mv "$LANG_DIR/$UNZIPPED_FOLDER"/*-ud-test.conllu "$LANG_DIR/test.conllu" 2>/dev/null

# Clean up the unzipped folder and other files
rm -rf "$LANG_DIR/$UNZIPPED_FOLDER"

echo "Success! Dataset for '$LANG_CODE' is ready in $LANG_DIR"
ls -l "$LANG_DIR"
