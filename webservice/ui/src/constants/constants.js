const RUN_MODE = "LOCAL"  // LOCAL or CLOUD
var MQTT_POSTFIX = "-3002.udacity-student-workspaces.com"
var CAMERA_FEED_POSTFIX = "-3004.udacity-student-workspaces.com"
var WS_HOST_START = window.location.hostname.split(".")[0].slice(0,-5)
if (RUN_MODE == "LOCAL") {
  WS_HOST_START = window.location.hostname.split(".")[0]
  MQTT_POSTFIX = ":3002"
  CAMERA_FEED_POSTFIX = ":3004"
}

export const WS_HOST = WS_HOST_START

export const SETTINGS = {
  CAMERA_FEED_SERVER: "http://" + WS_HOST + CAMERA_FEED_POSTFIX,
  CAMERA_FEED_WIDTH: 852,
  MAX_POINTS: 10,
  SLICE_LENGTH: -10,
};

export const LABELS = {
  START_TEXT: "Click me! ",
  END_TEXT: "The count is now: ",
};

export const HTTP = {
  CAMERA_FEED: `${SETTINGS.CAMERA_FEED_SERVER}/facstream.mjpeg`, // POST
};

export const MQTT = {
  MQTT_SERVER: "ws://" + WS_HOST + MQTT_POSTFIX,
  TOPICS: {
    PERSON: "person", // how many people did we see
    DURATION: "person/duration", // how long were they on frame
  },
};
