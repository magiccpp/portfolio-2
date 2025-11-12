latest_compare_result=$1
image_file=$2
receiver=$3


(
  echo "From: you@example.com"
  echo "To: xulilin20081@gmail.com"
  echo "Subject: Janus: Valar Morghulis, Valar Dohaeris."
  echo "MIME-Version: 1.0"
  echo "Content-Type: multipart/mixed; boundary=BOUNDARY"
  echo
  echo "--BOUNDARY"
  echo "Content-Type: text/plain; charset=UTF-8"
  echo
  cat "$latest_compare_result"
  echo
  echo "--BOUNDARY"
  echo "Content-Type: image/png"
  echo "Content-Transfer-Encoding: base64"
  echo "Content-Disposition: attachment; filename=\"output_chart.png\""
  echo
  base64 "$image_file"
  echo "--BOUNDARY--"
) | msmtp $receiver




