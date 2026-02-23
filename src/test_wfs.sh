curl -G "https://data.geopf.fr/wfs/ows" \
  --data-urlencode "SERVICE=WFS" \
  --data-urlencode "VERSION=1.1.0" \
  --data-urlencode "REQUEST=GetFeature" \
  --data-urlencode "TYPENAME=BDTOPO_V3:departement" \
  --data-urlencode "SRSNAME=EPSG:2154" \
  --data-urlencode "OUTPUTFORMAT=application/json" \
  --data-urlencode "BBOX=605000,6933000,606000,6934000,EPSG:2154" \
  | head -n 50