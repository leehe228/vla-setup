#!/usr/bin/env python3
#  Copyright (c) 2025 Franka Robotics GmbH
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, GroupAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node, PushRosNamespace
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    # --- 런치 인자 선언 --------------------------------------------------
    robot_ip_02   = LaunchConfiguration('robot_ip_02')
    robot_ip_03   = LaunchConfiguration('robot_ip_03')
    arm_id        = LaunchConfiguration('arm_id')
    load_gripper  = LaunchConfiguration('load_gripper')
    use_fake_hw   = LaunchConfiguration('use_fake_hardware')
    fake_sensors  = LaunchConfiguration('fake_sensor_commands')
    use_rviz      = LaunchConfiguration('use_rviz')

    declare_args = [
        DeclareLaunchArgument('robot_ip_02', default_value='172.16.0.2',
                              description='First arm IP'),
        DeclareLaunchArgument('robot_ip_03', default_value='172.16.0.3',
                              description='Second arm IP'),
        DeclareLaunchArgument('arm_id', default_value='fr3',
                              description='fr3 / fp3 / fer'),
        DeclareLaunchArgument('load_gripper', default_value='true',
                              description='attach gripper'),
        DeclareLaunchArgument('use_fake_hardware', default_value='false',
                              description='simulate hardware'),
        DeclareLaunchArgument('fake_sensor_commands', default_value='false',
                              description='if fake_hardware'),
        DeclareLaunchArgument('use_rviz', default_value='false',
                              description='launch RViz'),
    ]

    # --- dual_franka 런치 포함 --------------------------------------------
    include_dual = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare('franka_bringup'),
                'launch', 'dual_franka.launch.py'
            ])
        ]),
        launch_arguments={
            'robot_ip_02': robot_ip_02,
            'robot_ip_03': robot_ip_03,
            'arm_id':      arm_id,
            'load_gripper': load_gripper,
            'use_fake_hardware':     use_fake_hw,
            'fake_sensor_commands':  fake_sensors,
            'use_rviz':              use_rviz,
        }.items()
    )

    # --- 좌팔·우팔 teleop 노드 그룹 ---------------------------------------
    # (scripts/teleop_franka.py 가 franka_bringup 패키지 scripts/에 설치되어 있어야 합니다)
    left_teleop = GroupAction([
        PushRosNamespace('leftarm'),
        Node(
            package='franka_bringup',
            executable='teleop_franka.py',
            output='screen',
            parameters=[{'arm_ns': 'leftarm'}],
        ),
    ])
    right_teleop = GroupAction([
        PushRosNamespace('rightarm'),
        Node(
            package='franka_bringup',
            executable='teleop_franka.py',
            output='screen',
            parameters=[{'arm_ns': 'rightarm'}],
        ),
    ])

    return LaunchDescription(
        declare_args
        + [ include_dual, left_teleop, right_teleop ]
    )
