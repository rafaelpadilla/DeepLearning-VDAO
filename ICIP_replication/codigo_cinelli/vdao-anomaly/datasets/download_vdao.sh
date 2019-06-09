TARGET_DIR=$1
if [ -z $TARGET_DIR ]
then
  echo "Must specify target directory"
else
  mkdir $TARGET_DIR/
  URL=https://www.dropbox.com/s/39guyaakreu5o2r/vdao_feat_dataset.tar.gz?dl=0
  wget $URL -P $TARGET_DIR
  tar -xvf $TARGET_DIR/vdao_feat_dataset.tar.gz -C $TARGET_DIR
fi
