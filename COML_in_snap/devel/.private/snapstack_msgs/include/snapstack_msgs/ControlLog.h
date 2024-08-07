// Generated by gencpp from file snapstack_msgs/ControlLog.msg
// DO NOT EDIT!


#ifndef SNAPSTACK_MSGS_MESSAGE_CONTROLLOG_H
#define SNAPSTACK_MSGS_MESSAGE_CONTROLLOG_H


#include <string>
#include <vector>
#include <memory>

#include <ros/types.h>
#include <ros/serialization.h>
#include <ros/builtin_message_traits.h>
#include <ros/message_operations.h>

#include <std_msgs/Header.h>
#include <geometry_msgs/Vector3.h>
#include <geometry_msgs/Vector3.h>
#include <geometry_msgs/Vector3.h>
#include <geometry_msgs/Vector3.h>
#include <geometry_msgs/Vector3.h>
#include <geometry_msgs/Vector3.h>
#include <geometry_msgs/Vector3.h>
#include <geometry_msgs/Vector3.h>
#include <geometry_msgs/Vector3.h>
#include <geometry_msgs/Vector3.h>
#include <geometry_msgs/Vector3.h>
#include <geometry_msgs/Quaternion.h>
#include <geometry_msgs/Quaternion.h>
#include <geometry_msgs/Vector3.h>
#include <geometry_msgs/Vector3.h>
#include <geometry_msgs/Vector3.h>
#include <geometry_msgs/Vector3.h>
#include <geometry_msgs/Vector3.h>
#include <geometry_msgs/Vector3.h>

namespace snapstack_msgs
{
template <class ContainerAllocator>
struct ControlLog_
{
  typedef ControlLog_<ContainerAllocator> Type;

  ControlLog_()
    : header()
    , p()
    , p_ref()
    , p_err()
    , p_err_int()
    , v()
    , v_ref()
    , v_err()
    , a_ff()
    , a_fb()
    , j_ff()
    , j_fb()
    , q()
    , q_ref()
    , rpy()
    , rpy_ref()
    , w()
    , w_ref()
    , F_W()
    , P_norm(0.0)
    , A_norm(0.0)
    , y_norm(0.0)
    , f_hat()
    , power(false)  {
    }
  ControlLog_(const ContainerAllocator& _alloc)
    : header(_alloc)
    , p(_alloc)
    , p_ref(_alloc)
    , p_err(_alloc)
    , p_err_int(_alloc)
    , v(_alloc)
    , v_ref(_alloc)
    , v_err(_alloc)
    , a_ff(_alloc)
    , a_fb(_alloc)
    , j_ff(_alloc)
    , j_fb(_alloc)
    , q(_alloc)
    , q_ref(_alloc)
    , rpy(_alloc)
    , rpy_ref(_alloc)
    , w(_alloc)
    , w_ref(_alloc)
    , F_W(_alloc)
    , P_norm(0.0)
    , A_norm(0.0)
    , y_norm(0.0)
    , f_hat(_alloc)
    , power(false)  {
  (void)_alloc;
    }



   typedef  ::std_msgs::Header_<ContainerAllocator>  _header_type;
  _header_type header;

   typedef  ::geometry_msgs::Vector3_<ContainerAllocator>  _p_type;
  _p_type p;

   typedef  ::geometry_msgs::Vector3_<ContainerAllocator>  _p_ref_type;
  _p_ref_type p_ref;

   typedef  ::geometry_msgs::Vector3_<ContainerAllocator>  _p_err_type;
  _p_err_type p_err;

   typedef  ::geometry_msgs::Vector3_<ContainerAllocator>  _p_err_int_type;
  _p_err_int_type p_err_int;

   typedef  ::geometry_msgs::Vector3_<ContainerAllocator>  _v_type;
  _v_type v;

   typedef  ::geometry_msgs::Vector3_<ContainerAllocator>  _v_ref_type;
  _v_ref_type v_ref;

   typedef  ::geometry_msgs::Vector3_<ContainerAllocator>  _v_err_type;
  _v_err_type v_err;

   typedef  ::geometry_msgs::Vector3_<ContainerAllocator>  _a_ff_type;
  _a_ff_type a_ff;

   typedef  ::geometry_msgs::Vector3_<ContainerAllocator>  _a_fb_type;
  _a_fb_type a_fb;

   typedef  ::geometry_msgs::Vector3_<ContainerAllocator>  _j_ff_type;
  _j_ff_type j_ff;

   typedef  ::geometry_msgs::Vector3_<ContainerAllocator>  _j_fb_type;
  _j_fb_type j_fb;

   typedef  ::geometry_msgs::Quaternion_<ContainerAllocator>  _q_type;
  _q_type q;

   typedef  ::geometry_msgs::Quaternion_<ContainerAllocator>  _q_ref_type;
  _q_ref_type q_ref;

   typedef  ::geometry_msgs::Vector3_<ContainerAllocator>  _rpy_type;
  _rpy_type rpy;

   typedef  ::geometry_msgs::Vector3_<ContainerAllocator>  _rpy_ref_type;
  _rpy_ref_type rpy_ref;

   typedef  ::geometry_msgs::Vector3_<ContainerAllocator>  _w_type;
  _w_type w;

   typedef  ::geometry_msgs::Vector3_<ContainerAllocator>  _w_ref_type;
  _w_ref_type w_ref;

   typedef  ::geometry_msgs::Vector3_<ContainerAllocator>  _F_W_type;
  _F_W_type F_W;

   typedef float _P_norm_type;
  _P_norm_type P_norm;

   typedef float _A_norm_type;
  _A_norm_type A_norm;

   typedef float _y_norm_type;
  _y_norm_type y_norm;

   typedef  ::geometry_msgs::Vector3_<ContainerAllocator>  _f_hat_type;
  _f_hat_type f_hat;

   typedef uint8_t _power_type;
  _power_type power;





  typedef boost::shared_ptr< ::snapstack_msgs::ControlLog_<ContainerAllocator> > Ptr;
  typedef boost::shared_ptr< ::snapstack_msgs::ControlLog_<ContainerAllocator> const> ConstPtr;

}; // struct ControlLog_

typedef ::snapstack_msgs::ControlLog_<std::allocator<void> > ControlLog;

typedef boost::shared_ptr< ::snapstack_msgs::ControlLog > ControlLogPtr;
typedef boost::shared_ptr< ::snapstack_msgs::ControlLog const> ControlLogConstPtr;

// constants requiring out of line definition



template<typename ContainerAllocator>
std::ostream& operator<<(std::ostream& s, const ::snapstack_msgs::ControlLog_<ContainerAllocator> & v)
{
ros::message_operations::Printer< ::snapstack_msgs::ControlLog_<ContainerAllocator> >::stream(s, "", v);
return s;
}


template<typename ContainerAllocator1, typename ContainerAllocator2>
bool operator==(const ::snapstack_msgs::ControlLog_<ContainerAllocator1> & lhs, const ::snapstack_msgs::ControlLog_<ContainerAllocator2> & rhs)
{
  return lhs.header == rhs.header &&
    lhs.p == rhs.p &&
    lhs.p_ref == rhs.p_ref &&
    lhs.p_err == rhs.p_err &&
    lhs.p_err_int == rhs.p_err_int &&
    lhs.v == rhs.v &&
    lhs.v_ref == rhs.v_ref &&
    lhs.v_err == rhs.v_err &&
    lhs.a_ff == rhs.a_ff &&
    lhs.a_fb == rhs.a_fb &&
    lhs.j_ff == rhs.j_ff &&
    lhs.j_fb == rhs.j_fb &&
    lhs.q == rhs.q &&
    lhs.q_ref == rhs.q_ref &&
    lhs.rpy == rhs.rpy &&
    lhs.rpy_ref == rhs.rpy_ref &&
    lhs.w == rhs.w &&
    lhs.w_ref == rhs.w_ref &&
    lhs.F_W == rhs.F_W &&
    lhs.P_norm == rhs.P_norm &&
    lhs.A_norm == rhs.A_norm &&
    lhs.y_norm == rhs.y_norm &&
    lhs.f_hat == rhs.f_hat &&
    lhs.power == rhs.power;
}

template<typename ContainerAllocator1, typename ContainerAllocator2>
bool operator!=(const ::snapstack_msgs::ControlLog_<ContainerAllocator1> & lhs, const ::snapstack_msgs::ControlLog_<ContainerAllocator2> & rhs)
{
  return !(lhs == rhs);
}


} // namespace snapstack_msgs

namespace ros
{
namespace message_traits
{





template <class ContainerAllocator>
struct IsMessage< ::snapstack_msgs::ControlLog_<ContainerAllocator> >
  : TrueType
  { };

template <class ContainerAllocator>
struct IsMessage< ::snapstack_msgs::ControlLog_<ContainerAllocator> const>
  : TrueType
  { };

template <class ContainerAllocator>
struct IsFixedSize< ::snapstack_msgs::ControlLog_<ContainerAllocator> >
  : FalseType
  { };

template <class ContainerAllocator>
struct IsFixedSize< ::snapstack_msgs::ControlLog_<ContainerAllocator> const>
  : FalseType
  { };

template <class ContainerAllocator>
struct HasHeader< ::snapstack_msgs::ControlLog_<ContainerAllocator> >
  : TrueType
  { };

template <class ContainerAllocator>
struct HasHeader< ::snapstack_msgs::ControlLog_<ContainerAllocator> const>
  : TrueType
  { };


template<class ContainerAllocator>
struct MD5Sum< ::snapstack_msgs::ControlLog_<ContainerAllocator> >
{
  static const char* value()
  {
    return "f8bf2fc1a737a49fa227aa4e424a7184";
  }

  static const char* value(const ::snapstack_msgs::ControlLog_<ContainerAllocator>&) { return value(); }
  static const uint64_t static_value1 = 0xf8bf2fc1a737a49fULL;
  static const uint64_t static_value2 = 0xa227aa4e424a7184ULL;
};

template<class ContainerAllocator>
struct DataType< ::snapstack_msgs::ControlLog_<ContainerAllocator> >
{
  static const char* value()
  {
    return "snapstack_msgs/ControlLog";
  }

  static const char* value(const ::snapstack_msgs::ControlLog_<ContainerAllocator>&) { return value(); }
};

template<class ContainerAllocator>
struct Definition< ::snapstack_msgs::ControlLog_<ContainerAllocator> >
{
  static const char* value()
  {
    return "# The outer loop trajectory tracker generates this msg for analysis / debugging\n"
"\n"
"Header header\n"
"\n"
"# position signals\n"
"geometry_msgs/Vector3 p\n"
"geometry_msgs/Vector3 p_ref\n"
"geometry_msgs/Vector3 p_err\n"
"geometry_msgs/Vector3 p_err_int\n"
"\n"
"# velocity signals\n"
"geometry_msgs/Vector3 v\n"
"geometry_msgs/Vector3 v_ref\n"
"geometry_msgs/Vector3 v_err\n"
"\n"
"# acceleration signals\n"
"geometry_msgs/Vector3 a_ff\n"
"geometry_msgs/Vector3 a_fb\n"
"\n"
"# jerk signals\n"
"geometry_msgs/Vector3 j_ff\n"
"geometry_msgs/Vector3 j_fb\n"
"\n"
"# attitude signals\n"
"geometry_msgs/Quaternion q\n"
"geometry_msgs/Quaternion q_ref\n"
"geometry_msgs/Vector3 rpy\n"
"geometry_msgs/Vector3 rpy_ref\n"
"\n"
"# angular rate signals\n"
"geometry_msgs/Vector3 w\n"
"geometry_msgs/Vector3 w_ref\n"
"\n"
"geometry_msgs/Vector3 F_W # Desired total force [N], expressed in world\n"
"\n"
"float32 P_norm\n"
"float32 A_norm\n"
"float32 y_norm\n"
"geometry_msgs/Vector3 f_hat\n"
"\n"
"\n"
"bool power # true if motors should be able to spin\n"
"\n"
"# TODO: add outer (and inner?) parameters\n"
"================================================================================\n"
"MSG: std_msgs/Header\n"
"# Standard metadata for higher-level stamped data types.\n"
"# This is generally used to communicate timestamped data \n"
"# in a particular coordinate frame.\n"
"# \n"
"# sequence ID: consecutively increasing ID \n"
"uint32 seq\n"
"#Two-integer timestamp that is expressed as:\n"
"# * stamp.sec: seconds (stamp_secs) since epoch (in Python the variable is called 'secs')\n"
"# * stamp.nsec: nanoseconds since stamp_secs (in Python the variable is called 'nsecs')\n"
"# time-handling sugar is provided by the client library\n"
"time stamp\n"
"#Frame this data is associated with\n"
"string frame_id\n"
"\n"
"================================================================================\n"
"MSG: geometry_msgs/Vector3\n"
"# This represents a vector in free space. \n"
"# It is only meant to represent a direction. Therefore, it does not\n"
"# make sense to apply a translation to it (e.g., when applying a \n"
"# generic rigid transformation to a Vector3, tf2 will only apply the\n"
"# rotation). If you want your data to be translatable too, use the\n"
"# geometry_msgs/Point message instead.\n"
"\n"
"float64 x\n"
"float64 y\n"
"float64 z\n"
"================================================================================\n"
"MSG: geometry_msgs/Quaternion\n"
"# This represents an orientation in free space in quaternion form.\n"
"\n"
"float64 x\n"
"float64 y\n"
"float64 z\n"
"float64 w\n"
;
  }

  static const char* value(const ::snapstack_msgs::ControlLog_<ContainerAllocator>&) { return value(); }
};

} // namespace message_traits
} // namespace ros

namespace ros
{
namespace serialization
{

  template<class ContainerAllocator> struct Serializer< ::snapstack_msgs::ControlLog_<ContainerAllocator> >
  {
    template<typename Stream, typename T> inline static void allInOne(Stream& stream, T m)
    {
      stream.next(m.header);
      stream.next(m.p);
      stream.next(m.p_ref);
      stream.next(m.p_err);
      stream.next(m.p_err_int);
      stream.next(m.v);
      stream.next(m.v_ref);
      stream.next(m.v_err);
      stream.next(m.a_ff);
      stream.next(m.a_fb);
      stream.next(m.j_ff);
      stream.next(m.j_fb);
      stream.next(m.q);
      stream.next(m.q_ref);
      stream.next(m.rpy);
      stream.next(m.rpy_ref);
      stream.next(m.w);
      stream.next(m.w_ref);
      stream.next(m.F_W);
      stream.next(m.P_norm);
      stream.next(m.A_norm);
      stream.next(m.y_norm);
      stream.next(m.f_hat);
      stream.next(m.power);
    }

    ROS_DECLARE_ALLINONE_SERIALIZER
  }; // struct ControlLog_

} // namespace serialization
} // namespace ros

namespace ros
{
namespace message_operations
{

template<class ContainerAllocator>
struct Printer< ::snapstack_msgs::ControlLog_<ContainerAllocator> >
{
  template<typename Stream> static void stream(Stream& s, const std::string& indent, const ::snapstack_msgs::ControlLog_<ContainerAllocator>& v)
  {
    s << indent << "header: ";
    s << std::endl;
    Printer< ::std_msgs::Header_<ContainerAllocator> >::stream(s, indent + "  ", v.header);
    s << indent << "p: ";
    s << std::endl;
    Printer< ::geometry_msgs::Vector3_<ContainerAllocator> >::stream(s, indent + "  ", v.p);
    s << indent << "p_ref: ";
    s << std::endl;
    Printer< ::geometry_msgs::Vector3_<ContainerAllocator> >::stream(s, indent + "  ", v.p_ref);
    s << indent << "p_err: ";
    s << std::endl;
    Printer< ::geometry_msgs::Vector3_<ContainerAllocator> >::stream(s, indent + "  ", v.p_err);
    s << indent << "p_err_int: ";
    s << std::endl;
    Printer< ::geometry_msgs::Vector3_<ContainerAllocator> >::stream(s, indent + "  ", v.p_err_int);
    s << indent << "v: ";
    s << std::endl;
    Printer< ::geometry_msgs::Vector3_<ContainerAllocator> >::stream(s, indent + "  ", v.v);
    s << indent << "v_ref: ";
    s << std::endl;
    Printer< ::geometry_msgs::Vector3_<ContainerAllocator> >::stream(s, indent + "  ", v.v_ref);
    s << indent << "v_err: ";
    s << std::endl;
    Printer< ::geometry_msgs::Vector3_<ContainerAllocator> >::stream(s, indent + "  ", v.v_err);
    s << indent << "a_ff: ";
    s << std::endl;
    Printer< ::geometry_msgs::Vector3_<ContainerAllocator> >::stream(s, indent + "  ", v.a_ff);
    s << indent << "a_fb: ";
    s << std::endl;
    Printer< ::geometry_msgs::Vector3_<ContainerAllocator> >::stream(s, indent + "  ", v.a_fb);
    s << indent << "j_ff: ";
    s << std::endl;
    Printer< ::geometry_msgs::Vector3_<ContainerAllocator> >::stream(s, indent + "  ", v.j_ff);
    s << indent << "j_fb: ";
    s << std::endl;
    Printer< ::geometry_msgs::Vector3_<ContainerAllocator> >::stream(s, indent + "  ", v.j_fb);
    s << indent << "q: ";
    s << std::endl;
    Printer< ::geometry_msgs::Quaternion_<ContainerAllocator> >::stream(s, indent + "  ", v.q);
    s << indent << "q_ref: ";
    s << std::endl;
    Printer< ::geometry_msgs::Quaternion_<ContainerAllocator> >::stream(s, indent + "  ", v.q_ref);
    s << indent << "rpy: ";
    s << std::endl;
    Printer< ::geometry_msgs::Vector3_<ContainerAllocator> >::stream(s, indent + "  ", v.rpy);
    s << indent << "rpy_ref: ";
    s << std::endl;
    Printer< ::geometry_msgs::Vector3_<ContainerAllocator> >::stream(s, indent + "  ", v.rpy_ref);
    s << indent << "w: ";
    s << std::endl;
    Printer< ::geometry_msgs::Vector3_<ContainerAllocator> >::stream(s, indent + "  ", v.w);
    s << indent << "w_ref: ";
    s << std::endl;
    Printer< ::geometry_msgs::Vector3_<ContainerAllocator> >::stream(s, indent + "  ", v.w_ref);
    s << indent << "F_W: ";
    s << std::endl;
    Printer< ::geometry_msgs::Vector3_<ContainerAllocator> >::stream(s, indent + "  ", v.F_W);
    s << indent << "P_norm: ";
    Printer<float>::stream(s, indent + "  ", v.P_norm);
    s << indent << "A_norm: ";
    Printer<float>::stream(s, indent + "  ", v.A_norm);
    s << indent << "y_norm: ";
    Printer<float>::stream(s, indent + "  ", v.y_norm);
    s << indent << "f_hat: ";
    s << std::endl;
    Printer< ::geometry_msgs::Vector3_<ContainerAllocator> >::stream(s, indent + "  ", v.f_hat);
    s << indent << "power: ";
    Printer<uint8_t>::stream(s, indent + "  ", v.power);
  }
};

} // namespace message_operations
} // namespace ros

#endif // SNAPSTACK_MSGS_MESSAGE_CONTROLLOG_H
