while read p; do
  echo -e 1'\t'$p >> all_data.csv
done <quote.tok.gt9.5000

while read p; do
  echo -e 0'\t'$p >> all_data.csv
done <plot.tok.gt9.5000
