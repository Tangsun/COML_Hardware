// Auto-generated. Do not edit!

// (in-package snapstack_msgs.msg)


"use strict";

const _serializer = _ros_msg_utils.Serialize;
const _arraySerializer = _serializer.Array;
const _deserializer = _ros_msg_utils.Deserialize;
const _arrayDeserializer = _deserializer.Array;
const _finder = _ros_msg_utils.Find;
const _getByteLength = _ros_msg_utils.getByteLength;
let std_msgs = _finder('std_msgs');

//-----------------------------------------------------------

class TimeFilter {
  constructor(initObj={}) {
    if (initObj === null) {
      // initObj === null is a special case for deserialization where we don't initialize fields
      this.header = null;
      this.dt = null;
      this.delayed_dt = null;
      this.skipped = null;
      this.upper = null;
      this.lower = null;
    }
    else {
      if (initObj.hasOwnProperty('header')) {
        this.header = initObj.header
      }
      else {
        this.header = new std_msgs.msg.Header();
      }
      if (initObj.hasOwnProperty('dt')) {
        this.dt = initObj.dt
      }
      else {
        this.dt = 0.0;
      }
      if (initObj.hasOwnProperty('delayed_dt')) {
        this.delayed_dt = initObj.delayed_dt
      }
      else {
        this.delayed_dt = 0.0;
      }
      if (initObj.hasOwnProperty('skipped')) {
        this.skipped = initObj.skipped
      }
      else {
        this.skipped = false;
      }
      if (initObj.hasOwnProperty('upper')) {
        this.upper = initObj.upper
      }
      else {
        this.upper = 0.0;
      }
      if (initObj.hasOwnProperty('lower')) {
        this.lower = initObj.lower
      }
      else {
        this.lower = 0.0;
      }
    }
  }

  static serialize(obj, buffer, bufferOffset) {
    // Serializes a message object of type TimeFilter
    // Serialize message field [header]
    bufferOffset = std_msgs.msg.Header.serialize(obj.header, buffer, bufferOffset);
    // Serialize message field [dt]
    bufferOffset = _serializer.float32(obj.dt, buffer, bufferOffset);
    // Serialize message field [delayed_dt]
    bufferOffset = _serializer.float32(obj.delayed_dt, buffer, bufferOffset);
    // Serialize message field [skipped]
    bufferOffset = _serializer.bool(obj.skipped, buffer, bufferOffset);
    // Serialize message field [upper]
    bufferOffset = _serializer.float32(obj.upper, buffer, bufferOffset);
    // Serialize message field [lower]
    bufferOffset = _serializer.float32(obj.lower, buffer, bufferOffset);
    return bufferOffset;
  }

  static deserialize(buffer, bufferOffset=[0]) {
    //deserializes a message object of type TimeFilter
    let len;
    let data = new TimeFilter(null);
    // Deserialize message field [header]
    data.header = std_msgs.msg.Header.deserialize(buffer, bufferOffset);
    // Deserialize message field [dt]
    data.dt = _deserializer.float32(buffer, bufferOffset);
    // Deserialize message field [delayed_dt]
    data.delayed_dt = _deserializer.float32(buffer, bufferOffset);
    // Deserialize message field [skipped]
    data.skipped = _deserializer.bool(buffer, bufferOffset);
    // Deserialize message field [upper]
    data.upper = _deserializer.float32(buffer, bufferOffset);
    // Deserialize message field [lower]
    data.lower = _deserializer.float32(buffer, bufferOffset);
    return data;
  }

  static getMessageSize(object) {
    let length = 0;
    length += std_msgs.msg.Header.getMessageSize(object.header);
    return length + 17;
  }

  static datatype() {
    // Returns string type for a message object
    return 'snapstack_msgs/TimeFilter';
  }

  static md5sum() {
    //Returns md5sum for a message object
    return 'ac265f085f2218e2759385a5695df9af';
  }

  static messageDefinition() {
    // Returns full string definition for message
    return `
    # TimeFilter.msg
    
    Header header
    float32 dt  	# dt recorded in time stamp
    float32 delayed_dt  # dt stamped when arrived onboard
    bool skipped	# is msg skipped?
    float32 upper 	# upper bound
    float32 lower 	# lower bound
    ================================================================================
    MSG: std_msgs/Header
    # Standard metadata for higher-level stamped data types.
    # This is generally used to communicate timestamped data 
    # in a particular coordinate frame.
    # 
    # sequence ID: consecutively increasing ID 
    uint32 seq
    #Two-integer timestamp that is expressed as:
    # * stamp.sec: seconds (stamp_secs) since epoch (in Python the variable is called 'secs')
    # * stamp.nsec: nanoseconds since stamp_secs (in Python the variable is called 'nsecs')
    # time-handling sugar is provided by the client library
    time stamp
    #Frame this data is associated with
    string frame_id
    
    `;
  }

  static Resolve(msg) {
    // deep-construct a valid message object instance of whatever was passed in
    if (typeof msg !== 'object' || msg === null) {
      msg = {};
    }
    const resolved = new TimeFilter(null);
    if (msg.header !== undefined) {
      resolved.header = std_msgs.msg.Header.Resolve(msg.header)
    }
    else {
      resolved.header = new std_msgs.msg.Header()
    }

    if (msg.dt !== undefined) {
      resolved.dt = msg.dt;
    }
    else {
      resolved.dt = 0.0
    }

    if (msg.delayed_dt !== undefined) {
      resolved.delayed_dt = msg.delayed_dt;
    }
    else {
      resolved.delayed_dt = 0.0
    }

    if (msg.skipped !== undefined) {
      resolved.skipped = msg.skipped;
    }
    else {
      resolved.skipped = false
    }

    if (msg.upper !== undefined) {
      resolved.upper = msg.upper;
    }
    else {
      resolved.upper = 0.0
    }

    if (msg.lower !== undefined) {
      resolved.lower = msg.lower;
    }
    else {
      resolved.lower = 0.0
    }

    return resolved;
    }
};

module.exports = TimeFilter;
