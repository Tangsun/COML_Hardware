// Auto-generated. Do not edit!

// (in-package behavior_selector.srv)


"use strict";

const _serializer = _ros_msg_utils.Serialize;
const _arraySerializer = _serializer.Array;
const _deserializer = _ros_msg_utils.Deserialize;
const _arrayDeserializer = _deserializer.Array;
const _finder = _ros_msg_utils.Find;
const _getByteLength = _ros_msg_utils.getByteLength;

//-----------------------------------------------------------


//-----------------------------------------------------------

class MissionModeChangeRequest {
  constructor(initObj={}) {
    if (initObj === null) {
      // initObj === null is a special case for deserialization where we don't initialize fields
      this.mode = null;
    }
    else {
      if (initObj.hasOwnProperty('mode')) {
        this.mode = initObj.mode
      }
      else {
        this.mode = 0;
      }
    }
  }

  static serialize(obj, buffer, bufferOffset) {
    // Serializes a message object of type MissionModeChangeRequest
    // Serialize message field [mode]
    bufferOffset = _serializer.uint8(obj.mode, buffer, bufferOffset);
    return bufferOffset;
  }

  static deserialize(buffer, bufferOffset=[0]) {
    //deserializes a message object of type MissionModeChangeRequest
    let len;
    let data = new MissionModeChangeRequest(null);
    // Deserialize message field [mode]
    data.mode = _deserializer.uint8(buffer, bufferOffset);
    return data;
  }

  static getMessageSize(object) {
    return 1;
  }

  static datatype() {
    // Returns string type for a service object
    return 'behavior_selector/MissionModeChangeRequest';
  }

  static md5sum() {
    //Returns md5sum for a message object
    return '5c83eaecb198976dfe0f3ab2713b48c0';
  }

  static messageDefinition() {
    // Returns full string definition for message
    return `
    uint8 mode
    uint8 START  = 1
    uint8 END    = 2
    uint8 KILL   = 3
    
    `;
  }

  static Resolve(msg) {
    // deep-construct a valid message object instance of whatever was passed in
    if (typeof msg !== 'object' || msg === null) {
      msg = {};
    }
    const resolved = new MissionModeChangeRequest(null);
    if (msg.mode !== undefined) {
      resolved.mode = msg.mode;
    }
    else {
      resolved.mode = 0
    }

    return resolved;
    }
};

// Constants for message
MissionModeChangeRequest.Constants = {
  START: 1,
  END: 2,
  KILL: 3,
}

class MissionModeChangeResponse {
  constructor(initObj={}) {
    if (initObj === null) {
      // initObj === null is a special case for deserialization where we don't initialize fields
      this.RECEIVED = null;
    }
    else {
      if (initObj.hasOwnProperty('RECEIVED')) {
        this.RECEIVED = initObj.RECEIVED
      }
      else {
        this.RECEIVED = false;
      }
    }
  }

  static serialize(obj, buffer, bufferOffset) {
    // Serializes a message object of type MissionModeChangeResponse
    // Serialize message field [RECEIVED]
    bufferOffset = _serializer.bool(obj.RECEIVED, buffer, bufferOffset);
    return bufferOffset;
  }

  static deserialize(buffer, bufferOffset=[0]) {
    //deserializes a message object of type MissionModeChangeResponse
    let len;
    let data = new MissionModeChangeResponse(null);
    // Deserialize message field [RECEIVED]
    data.RECEIVED = _deserializer.bool(buffer, bufferOffset);
    return data;
  }

  static getMessageSize(object) {
    return 1;
  }

  static datatype() {
    // Returns string type for a service object
    return 'behavior_selector/MissionModeChangeResponse';
  }

  static md5sum() {
    //Returns md5sum for a message object
    return 'deaed58dd82a93233dc3cd85c4d94e40';
  }

  static messageDefinition() {
    // Returns full string definition for message
    return `
    bool RECEIVED
    
    `;
  }

  static Resolve(msg) {
    // deep-construct a valid message object instance of whatever was passed in
    if (typeof msg !== 'object' || msg === null) {
      msg = {};
    }
    const resolved = new MissionModeChangeResponse(null);
    if (msg.RECEIVED !== undefined) {
      resolved.RECEIVED = msg.RECEIVED;
    }
    else {
      resolved.RECEIVED = false
    }

    return resolved;
    }
};

module.exports = {
  Request: MissionModeChangeRequest,
  Response: MissionModeChangeResponse,
  md5sum() { return '485ed44d9d9a6cafffa6f772d726a264'; },
  datatype() { return 'behavior_selector/MissionModeChange'; }
};
