scanloopdb=../scanloopdb
looplen=20
dbfile=../loops_full.db
bounds="H95 H102"
nhits="-n 20"

$scanloopdb -l $looplen $nhits -r $bounds $dbfile pdb1yqv.ent 1yqv_${looplen}.hits



