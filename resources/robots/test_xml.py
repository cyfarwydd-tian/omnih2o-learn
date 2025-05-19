import mujoco
import glfw
import sys

# Set the path to your MJCF XML file here
xml_path = "g1/g1_29Dof.xml"
#xml_path = "g1_description/h1.xml"

# Load the MJCF model
model = mujoco.MjModel.from_xml_path(xml_path)
data = mujoco.MjData(model)

# Initialize GLFW window
if not glfw.init():
    print("Could not initialize GLFW")
    sys.exit(1)

window = glfw.create_window(1200, 900, "MuJoCo Viewer", None, None)
if not window:
    glfw.terminate()
    print("Could not create window")
    sys.exit(1)

glfw.make_context_current(window)

# Create the scene and context for rendering
scene = mujoco.MjvScene(model, maxgeom=10000)
context = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150.value)

# Set up camera and rendering options
cam = mujoco.MjvCamera()
opt = mujoco.MjvOption()
mujoco.mjv_defaultCamera(cam)
mujoco.mjv_defaultOption(opt)

# Configure initial camera view
cam.azimuth = 90.0        # Horizontal rotation
cam.elevation = -20.0     # Vertical angle
cam.distance = 3.0        # Zoom distance
cam.lookat = [0, 0, 1.0]  # Focus point

# Main rendering loop
while not glfw.window_should_close(window):
    # Step the simulation (no control applied, passive visualization)
    mujoco.mj_step(model, data)

    # Update scene with current model state
    mujoco.mjv_updateScene(model, data, opt, None, cam, mujoco.mjtCatBit.mjCAT_ALL.value, scene)

    # Get current window size and set viewport
    viewport_width, viewport_height = glfw.get_framebuffer_size(window)
    viewport = mujoco.MjrRect(0, 0, viewport_width, viewport_height)

    # Render the scene
    mujoco.mjr_render(viewport, scene, context)

    # Handle window events
    glfw.swap_buffers(window)
    glfw.poll_events()

# Clean up after window is closed
glfw.terminate()
