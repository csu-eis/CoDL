## Backup
Before backup, you should check that directory `mace/codegen/models` is deleted. This directory contrains large size model file which will make your backup package file also be very large, so I recommand you delete it.
```shell
cd mace-master/

# Delete model directory
rm -rf mace/codegen/models/

# Run backup script
sh tools/mace-backup-fucheng-script.sh
```