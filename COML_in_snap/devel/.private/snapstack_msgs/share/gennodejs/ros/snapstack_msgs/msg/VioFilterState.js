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
let geometry_msgs = _finder('geometry_msgs');

//-----------------------------------------------------------

class VioFilterState {
  constructor(initObj={}) {
    if (initObj === null) {
      // initObj === null is a special case for deserialization where we don't initialize fields
      this.header = null;
      this.pose = null;
      this.twist = null;
      this.bw = null;
      this.ba = null;
      this.extrinsics = null;
      this.accel_meas = null;
      this.N = null;
      this.error_cov = null;
    }
    else {
      if (initObj.hasOwnProperty('header')) {
        this.header = initObj.header
      }
      else {
        this.header = new std_msgs.msg.Header();
      }
      if (initObj.hasOwnProperty('pose')) {
        this.pose = initObj.pose
      }
      else {
        this.pose = new geometry_msgs.msg.Pose();
      }
      if (initObj.hasOwnProperty('twist')) {
        this.twist = initObj.twist
      }
      else {
        this.twist = new geometry_msgs.msg.Twist();
      }
      if (initObj.hasOwnProperty('bw')) {
        this.bw = initObj.bw
      }
      else {
        this.bw = new geometry_msgs.msg.Vector3();
      }
      if (initObj.hasOwnProperty('ba')) {
        this.ba = initObj.ba
      }
      else {
        this.ba = new geometry_msgs.msg.Vector3();
      }
      if (initObj.hasOwnProperty('extrinsics')) {
        this.extrinsics = initObj.extrinsics
      }
      else {
        this.extrinsics = new geometry_msgs.msg.Pose();
      }
      if (initObj.hasOwnProperty('accel_meas')) {
        this.accel_meas = initObj.accel_meas
      }
      else {
        this.accel_meas = new geometry_msgs.msg.Vector3();
      }
      if (initObj.hasOwnProperty('N')) {
        this.N = initObj.N
      }
      else {
        this.N = 0;
      }
      if (initObj.hasOwnProperty('error_cov')) {
        this.error_cov = initObj.error_cov
      }
      else {
        this.error_cov = [];
      }
    }
  }

  static serialize(obj, buffer, bufferOffset) {
    // Serializes a message object of type VioFilterState
    // Serialize message field [header]
    bufferOffset = std_msgs.msg.Header.serialize(obj.header, buffer, bufferOffset);
    // Serialize message field [pose]
    bufferOffset = geometry_msgs.msg.Pose.serialize(obj.pose, buffer, bufferOffset);
    // Serialize message field [twist]
    bufferOffset = geometry_msgs.msg.Twist.serialize(obj.twist, buffer, bufferOffset);
    // Serialize message field [bw]
    bufferOffset = geometry_msgs.msg.Vector3.serialize(obj.bw, buffer, bufferOffset);
    // Serialize message field [ba]
    bufferOffset = geometry_msgs.msg.Vector3.serialize(obj.ba, buffer, bufferOffset);
    // Serialize message field [extrinsics]
    bufferOffset = geometry_msgs.msg.Pose.serialize(obj.extrinsics, buffer, bufferOffset);
    // Serialize message field [accel_meas]
    bufferOffset = geometry_msgs.msg.Vector3.serialize(obj.accel_meas, buffer, bufferOffset);
    // Serialize message field [N]
    bufferOffset = _serializer.int32(obj.N, buffer, bufferOffset);
    // Serialize message field [error_cov]
    bufferOffset = _arraySerializer.float32(obj.error_cov, buffer, bufferOffset, null);
    return bufferOffset;
  }

  static deserialize(buffer, bufferOffset=[0]) {
    //deserializes a message object of type VioFilterState
    let len;
    let data = new VioFilterState(null);
    // Deserialize message field [header]
    data.header = std_msgs.msg.Header.deserialize(buffer, bufferOffset);
    // Deserialize message field [pose]
    data.pose = geometry_msgs.msg.Pose.deserialize(buffer, bufferOffset);
    // Deserialize message field [twist]
    data.twist = geometry_msgs.msg.Twist.deserialize(buffer, bufferOffset);
    // Deserialize message field [bw]
    data.bw = geometry_msgs.msg.Vector3.deserialize(buffer, bufferOffset);
    // Deserialize message field [ba]
    data.ba = geometry_msgs.msg.Vector3.deserialize(buffer, bufferOffset);
    // Deserialize message field [extrinsics]
    data.extrinsics = geometry_msgs.msg.Pose.deserialize(buffer, bufferOffset);
    // Deserialize message field [accel_meas]
    data.accel_meas = geometry_msgs.msg.Vector3.deserialize(buffer, bufferOffset);
    // Deserialize message field [N]
    data.N = _deserializer.int32(buffer, bufferOffset);
    // Deserialize message field [error_cov]
    data.error_cov = _arrayDeserializer.float32(buffer, bufferOffset, null)
    return data;
  }

  static getMessageSize(object) {
    let length = 0;
    length += std_msgs.msg.Header.getMessageSize(object.header);
    length += 4 * object.error_cov.length;
    return length + 240;
  }

  static datatype() {
    // Returns string type for a message object
    return 'snapstack_msgs/VioFilterState';
  }

  static md5sum() {
    //Returns md5sum for a message object
    return '081bdc0c4d73ef878229dd1b85815934';
  }

  static messageDefinition() {
    // Returns full string definition for message
    return `
    # VioFilterState.msg
    std_msgs/Header header
    
    geometry_msgs/Pose pose
    geometry_msgs/Twist twist  # includes gyro measurement for rates
    geometry_msgs/Vector3 bw
    geometry_msgs/Vector3 ba
    geometry_msgs/Pose extrinsics  # camera-to-IMU transform
    
    geometry_msgs/Vector3 accel_meas  # (IMU-frame) raw accel measurement
    
    int32 N  # error state dimension
    float32[] error_cov  # NxN error covariance.
                         # For consistency, leading 21 terms are ordered [pos, vel, Rwb, bw, ba, Rcb, tcb].
    
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
    
    ================================================================================
    MSG: geometry_msgs/Pose
    # A representation of pose in free space, composed of position and orientation. 
    Point position
    Quaternion orientation
    
    ================================================================================
    MSG: geometry_msgs/Point
    # This contains the position of a point in free space
    float64 x
    float64 y
    float64 z
    
    ================================================================================
    MSG: geometry_msgs/Quaternion
    # This represents an orientation in free space in quaternion form.
    
    float64 x
    float64 y
    float64 z
    float64 w
    
    ================================================================================
    MSG: geometry_msgs/Twist
    # This expresses velocity in free space broken into its linear and angular parts.
    Vector3  linear
    Vector3  angular
    
    ================================================================================
    MSG: geometry_msgs/Vector3
    # This represents a vector in free space. 
    # It is only meant to represent a direction. Therefore, it does not
    # make sense to apply a translation to it (e.g., when applying a 
    # generic rigid transformation to a Vector3, tf2 will only apply the
    # rotation). If you want your data to be translatable too, use the
    # geometry_msgs/Point message instead.
    
    float64 x
    float64 y
    float64 z
    `;
  }

  static Resolve(msg) {
    // deep-construct a valid message object instance of whatever was passed in
    if (typeof msg !== 'object' || msg === null) {
      msg = {};
    }
    const resolved = new VioFilterState(null);
    if (msg.header !== undefined) {
      resolved.header = std_msgs.msg.Header.Resolve(msg.header)
    }
    else {
      resolved.header = new std_msgs.msg.Header()
    }

    if (msg.pose !== undefined) {
      resolved.pose = geometry_msgs.msg.Pose.Resolve(msg.pose)
    }
    else {
      resolved.pose = new geometry_msgs.msg.Pose()
    }

    if (msg.twist !== undefined) {
      resolved.twist = geometry_msgs.msg.Twist.Resolve(msg.twist)
    }
    else {
      resolved.twist = new geometry_msgs.msg.Twist()
    }

    if (msg.bw !== undefined) {
      resolved.bw = geometry_msgs.msg.Vector3.Resolve(msg.bw)
    }
    else {
      resolved.bw = new geometry_msgs.msg.Vector3()
    }

    if (msg.ba !== undefined) {
      resolved.ba = geometry_msgs.msg.Vector3.Resolve(msg.ba)
    }
    else {
      resolved.ba = new geometry_msgs.msg.Vector3()
    }

    if (msg.extrinsics !== undefined) {
      resolved.extrinsics = geometry_msgs.msg.Pose.Resolve(msg.extrinsics)
    }
    else {
      resolved.extrinsics = new geometry_msgs.msg.Pose()
    }

    if (msg.accel_meas !== undefined) {
      resolved.accel_meas = geometry_msgs.msg.Vector3.Resolve(msg.accel_meas)
    }
    else {
      resolved.accel_meas = new geometry_msgs.msg.Vector3()
    }

    if (msg.N !== undefined) {
      resolved.N = msg.N;
    }
    else {
      resolved.N = 0
    }

    if (msg.error_cov !== undefined) {
      resolved.error_cov = msg.error_cov;
    }
    else {
      resolved.error_cov = []
    }

    return resolved;
    }
};

module.exports = VioFilterState;
