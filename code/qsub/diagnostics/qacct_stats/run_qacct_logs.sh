#!/bin/bash
# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
# Saves SGE job accounting data to CSV
#
# Usage:
#   ./run_qacct_logs.sh username days
#   ./run_qacct_logs.sh fache 7
# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

USER=$1
DAYS=$2
DATE=$(date +%Y%m%d)
OUTFILE="qacct_${USER}_${DAYS}D_${DATE}.csv"

echo "[running] qacct -o $USER -j -d $DAYS > $OUTFILE"

# write header
echo "jobnumber,taskid,jobname,owner,group,project,department,account,priority,qname,hostname,granted_pe,slots,qsub_time,start_time,end_time,ru_wallclock,ru_utime,ru_stime,ru_maxrss,ru_ixrss,ru_ismrss,ru_idrss,ru_isrss,ru_minflt,ru_majflt,ru_nswap,ru_inblock,ru_oublock,ru_msgsnd,ru_msgrcv,ru_nsignals,ru_nvcsw,ru_nivcsw,cpu,mem,io,iow,maxvmem,arid,exit_status,failed" > $OUTFILE

# parse all fields — one row per job
qacct -o $USER -j -d $DAYS | awk '
/^jobnumber/    {jn=$2}
/^taskid/       {ti=$2}
/^jobname/      {nm=$2}
/^owner/        {ow=$2}
/^group/        {gr=$2}
/^project/      {pj=$2}
/^department/   {dp=$2}
/^account/      {ac=$2}
/^priority/     {pr=$2}
/^qname/        {qn=$2}
/^hostname/     {hn=$2}
/^granted_pe/   {gp=$2}
/^slots/        {sl=$2}
/^qsub_time/    {qt=$0; sub(/qsub_time[[:space:]]*/,"",qt)}
/^start_time/   {stt=$0; sub(/start_time[[:space:]]*/,"",stt)}
/^end_time/     {et=$0; sub(/end_time[[:space:]]*/,"",et)}
/^ru_wallclock/ {wc=$2}
/^ru_utime/     {ut=$2}
/^ru_stime/     {st=$2}
/^ru_maxrss/    {mrss=$2}
/^ru_ixrss/     {ixrss=$2}
/^ru_ismrss/    {ismrss=$2}
/^ru_idrss/     {idrss=$2}
/^ru_isrss/     {isrss=$2}
/^ru_minflt/    {mf=$2}
/^ru_majflt/    {mj=$2}
/^ru_nswap/     {ns=$2}
/^ru_inblock/   {ib=$2}
/^ru_oublock/   {ob=$2}
/^ru_msgsnd/    {ms=$2}
/^ru_msgrcv/    {mc=$2}
/^ru_nsignals/  {sig=$2}
/^ru_nvcsw/     {nv=$2}
/^ru_nivcsw/    {ni=$2}
/^cpu/          {cp=$2}
/^mem/          {me=$2}
/^io/           {io=$2}
/^iow/          {iow=$2}
/^maxvmem/      {mv=$2}
/^arid/         {ar=$2}
/^exit_status/  {es=$2}
/^failed/       {fa=$2;
  print jn","ti","nm","ow","gr","pj","dp","ac","pr","qn","hn","gp","sl","qt","stt","et","wc","ut","st","mrss","ixrss","ismrss","idrss","isrss","mf","mj","ns","ib","ob","ms","mc","sig","nv","ni","cp","me","io","iow","mv","ar","es","fa
}
' >> $OUTFILE

echo "Done. $(( $(wc -l < $OUTFILE) - 1 )) jobs written to $OUTFILE"