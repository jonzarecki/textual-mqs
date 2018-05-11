while read p; do
  echo -e 1'\t'$p >> all_data.csv
done <rt-polarity.pos

while read p; do
  echo -e 0'\t'$p >> all_data.csv
done <rt-polarity.neg
