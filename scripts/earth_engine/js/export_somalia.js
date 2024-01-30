// Script to export 145x145 pixel tiles from Google Earth Engine.
// Upload the script and the SO_Admin2_1990 shapefiles in GEE, run, wait, and press the
// 'RUN' buttons in the Tasks tab to start the export tasks.
// You need a Google Cloud Storage bucket. Replace the username and bucket name (search for todo)

// The first IPC scores were collected every 3 months, the second set was collected every 4 months.
var m3startDate = '2013-05-01';
var m3startDateObj = ee.Date(m3startDate);
var m3nMonths = 30;
var m4startDate = '2015-11-01'
var m4startDateObj = ee.Date(m4startDate);
var m4nMonths = 52;

var m3collection = ee.ImageCollection('LANDSAT/LC08/C02/T1_L2')
  .filterDate(m3startDate, '2015-11-01');
var m4collection = ee.ImageCollection('LANDSAT/LC08/C02/T1_L2')
  .filterDate(m4startDate, '2020-03-01');

// todo replace username
var shp = ee.FeatureCollection('users/<username>/SO_Admin2_1990');
print('shp: ', shp);

m3collection = m3collection.filterBounds(shp.geometry());
print('m3collection: ', m3collection);
m4collection = m4collection.filterBounds(shp.geometry());

// Applies scaling factors.
function applyScaleFactors(image) {
  var opticalBands = image.select('SR_B.').multiply(0.0000275).add(-0.2);
  return image.addBands(opticalBands, null, true);
}

m3collection = m3collection.map(applyScaleFactors);
print('m3collection: ', m3collection);
m4collection = m4collection.map(applyScaleFactors);

// Mask clouded pixels
function maskFn(image) {
  // Develop masks for unwanted pixels (fill, cloud, cloud shadow).
  var qaMask = image.select('QA_PIXEL').bitwiseAnd(parseInt('11111', 2)).eq(0);
  var saturationMask = image.select('QA_RADSAT').eq(0);
  return image.updateMask(qaMask).updateMask(saturationMask);
}

m3collection = m3collection.map(maskFn);
print('m3collection: ', m3collection);
m4collection = m4collection.map(maskFn);


m3collection = m3collection.select('SR_B.');
m4collection = m4collection.select('SR_B.');
print('m3collection selected: ', m3collection);
var castList = ['SR_B1', 'SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B6', 'SR_B7'];
var castDict = {
  'SR_B1': 'float',
  'SR_B2': 'float',
  'SR_B3': 'float',
  'SR_B4': 'float',
  'SR_B5': 'float',
  'SR_B6': 'float',
  'SR_B7': 'float',
};
m3collection = m3collection.cast(castDict, castList);
m4collection = m4collection.cast(castDict, castList);
print('m3collection casted: ', m3collection);


print('shp.geometry: ', shp.geometry());
print('shp.first().geometry: ', shp.first().geometry());


var m3seq = ee.List.sequence(0, m3nMonths-1, 3);
print(m3seq);
var m4seq = ee.List.sequence(0, m4nMonths-1, 4);

// Make composites
// https://gis.stackexchange.com/questions/340696/how-to-make-seasonal-landsat-tm-image-composites-in-google-earth-engine
function compFn(col, startDate, size) {
  return function(month) {
    month = ee.Number(month);
    var t1 = startDate.advance(month, 'month');
    var t2 = t1.advance(size, 'month');
    var imgMed = col.filterDate(t1, t2).median();
    var nBands = imgMed.bandNames().size();
    return imgMed.set({
      'time_start': t1.format('YYYY-MM-dd'),
      'time_end': t2.format('YYYY-MM-dd'),
      'nBands': nBands});
  }
}

var comp3 = ee.ImageCollection(m3seq.map(compFn(m3collection, m3startDateObj, 3)));
print('comp3 len: ', comp3.size());
print('comp3: ', comp3.limit(5));
print('comp3 len: ', comp3.size());
var comp4 = ee.ImageCollection(m4seq.map(compFn(m4collection, m4startDateObj, 4)));

// Remove images w/ no bands (can happen if there were no images for a date range).
comp3 = comp3.filter(ee.Filter.gt('nBands', 0));
comp4 = comp4.filter(ee.Filter.gt('nBands', 0));

// Check if projection is still the same
var projection = m3collection.first().select('SR_B2').projection().getInfo();
print('projection: ', projection);
var projections = m3collection.toList(m3collection.size()).map(function(img) {
  return ee.Feature(ee.Image(img).select('SR_B2').projection());
})
print('projections: ', projections);

projection = comp3.first().select('SR_B2').projection().getInfo();
print('projectionComp: ', projection);
projections = comp3.toList(comp3.size()).map(function(img) {
  return ee.Feature(ee.Image(img).select('SR_B2').projection());
})
print('projectionsComp: ', projections);

// Visualize on the GEE console map
var visualization = {
  bands: ['SR_B4', 'SR_B3', 'SR_B2'],
  min: 0.0,
  max: 0.3,
};

Map.setCenter(46.2, 4.15, 5);
var filtered2 = comp3.limit(2);
Map.addLayer(comp3, visualizaton, 'm3perRegion');
print('name: ', 'somalia' + '_' + filtered2.first().get('time_start').getInfo())

// Export to cloud
var wrap = function(img, n, monthstr) {
  var proj = img.select('SR_B2').projection().getInfo();
  var desc = 'somalia_' + monthstr + '_' + img.get('time_start').getInfo() + '_';
  Export.image.toCloudStorage({
    image: img,
    description: desc,
    crs: proj.crs,
    scale: 30,
    fileNamePrefix: desc,
    skipEmptyTiles: true,
    shardSize: 145,
    fileDimensions: 145,
    // todo replace bucket name
    bucket: '<bucket_name>',
    // https://gis.stackexchange.com/questions/278845/error-in-export-image-from-google-earth-engine
    region: shp.geometry(),
    maxPixels: 1e13,
  });
}

var colList = comp3.toList(comp3.size());
var i = 0
while (i >= 0) {
  try {
    var img = ee.Image(colList.get(i));
    wrap(img, i, 'm3')
    i++
  } catch (err) {
    var msg = err.message
    if (msg.slice(0, 36) === 'List.get: List index must be between') {
      break
    } else {
      print(msg)
      break
    }
  }
}
var colList4 = comp4.toList(comp4.size());
i = 0
while (i >= 0) {
  try {
    img = ee.Image(colList4.get(i));
    wrap(img, i, 'm4')
    i++
  } catch (err) {
    msg = err.message
    if (msg.slice(0, 36) === 'List.get: List index must be between') {
      break
    } else {
      print(msg)
      break
    }
  }
}
