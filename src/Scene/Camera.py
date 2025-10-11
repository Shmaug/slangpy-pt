import slangpy as spy

class Camera:
    def __init__(self):
        super().__init__()
        self.width = 100
        self.height = 100
        self.aspect_ratio = 1.0
        self.position = spy.float3(1, 1, 1)
        self.target = spy.float3(0, 0, 0)
        self.up = spy.float3(0, 1, 0)
        self.fov = 70.0
        self.recompute()

    def recompute(self):
        self.aspect_ratio = float(self.width) / float(self.height)

        self.fwd = spy.math.normalize(self.target - self.position)
        self.right = spy.math.normalize(spy.math.cross(self.fwd, self.up))
        self.up = spy.math.normalize(spy.math.cross(self.right, self.fwd))

        fov = spy.math.radians(self.fov)

        self.image_u = self.right * spy.math.tan(fov * 0.5) * self.aspect_ratio
        self.image_v = self.up * spy.math.tan(fov * 0.5)
        self.image_w = self.fwd

    def parameters(self):
        return {
            "position": self.position,
            "image_u": self.image_u,
            "image_v": self.image_v,
            "image_w": self.image_w
        }

class CameraController:
    MOVE_KEYS = {
        spy.KeyCode.a: spy.float3(-1, 0, 0),
        spy.KeyCode.d: spy.float3(1, 0, 0),
        spy.KeyCode.e: spy.float3(0, 1, 0),
        spy.KeyCode.q: spy.float3(0, -1, 0),
        spy.KeyCode.w: spy.float3(0, 0, 1),
        spy.KeyCode.s: spy.float3(0, 0, -1),
    }
    MOVE_SHIFT_FACTOR = 10.0

    def __init__(self, camera: Camera):
        super().__init__()
        self.camera = camera
        self.mouse_down = False
        self.mouse_pos = spy.float2()
        self.key_state = {k: False for k in CameraController.MOVE_KEYS.keys()}
        self.shift_down = False

        self.move_delta = spy.float3()
        self.rotate_delta = spy.float2()

        self.move_speed = 1.0
        self.rotate_speed = 0.002

    def update(self, dt: float):
        changed = False
        position = self.camera.position
        fwd = self.camera.fwd
        up = self.camera.up
        right = self.camera.right

        # Move
        if spy.math.length(self.move_delta) > 0:
            offset = right * self.move_delta.x
            offset += up * self.move_delta.y
            offset += fwd * self.move_delta.z
            factor = CameraController.MOVE_SHIFT_FACTOR if self.shift_down else 1.0
            offset *= self.move_speed * factor * dt
            position += offset
            changed = True

        # Rotate
        if spy.math.length(self.rotate_delta) > 0:
            yaw = spy.math.atan2(fwd.z, fwd.x)
            pitch = spy.math.asin(fwd.y)
            yaw += self.rotate_speed * self.rotate_delta.x
            pitch -= self.rotate_speed * self.rotate_delta.y
            fwd = spy.float3(
                spy.math.cos(yaw) * spy.math.cos(pitch),
                spy.math.sin(pitch),
                spy.math.sin(yaw) * spy.math.cos(pitch),
            )
            self.rotate_delta = spy.float2()
            changed = True

        if changed:
            self.camera.position = position
            self.camera.target = position + fwd
            self.camera.up = spy.float3(0, 1, 0)
            self.camera.recompute()

        return changed

    def on_keyboard_event(self, event: spy.KeyboardEvent):
        if event.is_key_press() or event.is_key_release():
            down = event.is_key_press()
            if event.key in CameraController.MOVE_KEYS:
                self.key_state[event.key] = down
            elif event.key == spy.KeyCode.left_shift:
                self.shift_down = down
        self.move_delta = spy.float3()
        for key, state in self.key_state.items():
            if state:
                self.move_delta += CameraController.MOVE_KEYS[key]

    def on_mouse_event(self, event: spy.MouseEvent):
        self.rotate_delta = spy.float2()
        if event.is_button_down() and event.button == spy.MouseButton.left:
            self.mouse_down = True
        if event.is_button_up() and event.button == spy.MouseButton.left:
            self.mouse_down = False
        if event.is_move():
            mouse_delta = event.pos - self.mouse_pos
            if self.mouse_down:
                self.rotate_delta = mouse_delta
            self.mouse_pos = event.pos
