import redis
import json
import os

class RedisClient:
    def __init__(self):
        # Tải cấu hình từ biến môi trường
        self.host = os.getenv("REDIS_HOST", "127.0.0.1")
        self.port = int(os.getenv("REDIS_PORT", 6379))
        self.client = None
        self._connect()  # Tự động kết nối khi khởi tạo
        self._setup_events()

    def _connect(self):

        """
        Kết nối với Redis server.
        """
        try:
            self.client = redis.Redis(host=self.host, port=self.port, decode_responses=True)
            self.client.ping()

            print("✅ Redis connected!")
        except redis.ConnectionError as error:
            print("❌ Redis connection failed:", error)

    def _setup_events(self):
        """
        Định nghĩa các sự kiện cho Redis (nếu có). 
        Python Redis không cung cấp sự kiện native, bạn cần thêm log nếu cần.
        """
        pass

    async def reload(self):
        """
        Kết nối lại với Redis server.
        """
        try:
            print("🔄 Reloading Redis connection...")
            self.disconnect()  # Ngắt kết nối trước đó
            self._connect()    # Kết nối lại
            print("✅ Redis reloaded successfully.")
        except Exception as err:
            print("❌ Error reloading Redis:", err)

    def disconnect(self):
        """
        Ngắt kết nối Redis.
        """
        try:
            if self.client:
                self.client.close()
                print("🔌 Redis client disconnected.")
        except Exception as err:
            print("❌ Error disconnecting Redis:", err)

    # CRUD OPERATIONS
    def set(self, key, value, expiration_in_seconds=3600):
   
        try:
        # Chuyển đổi datetime thành chuỗi ISO 8601
            if isinstance(value, list):
                self.client.setex(key, expiration_in_seconds, json.dumps(value))
                print(f"✅ Key {key} set with expiration {expiration_in_seconds} seconds")
        except Exception as err:
            print(f"❌ Error setting key {key}:", err)


    def get(self, key):
        """
        Lấy giá trị từ Redis.
        """
        try:
            value = self.client.get(key)
            return json.loads(value) if value else None
        except Exception as err:
            print(f"❌ Error getting key {key}:", err)
            return None

    def delete(self, key):
        """
        Xóa một khóa từ Redis.
        """
        try:
            self.client.delete(key)
            print(f"✅ Key {key} deleted")
        except Exception as err:
            print(f"❌ Error deleting key {key}:", err)

    def exists(self, key):
        """
        Kiểm tra sự tồn tại của một khóa trong Redis.
        """
        try:
            return self.client.exists(key) == 1
        except Exception as err:
            print(f"❌ Error checking existence for key {key}:", err)
            return False

    def keys(self, pattern="*"):
        """
        Lấy danh sách các khóa từ Redis theo pattern.
        """
        try:
            keys = self.client.keys(pattern)
            print(f"✅ Found {len(keys)} keys matching pattern '{pattern}'")
            return keys
        except Exception as err:
            print(f"❌ Error fetching keys with pattern {pattern}:", err)
            return []

    # ADVANCED OPERATIONS
    def sadd(self, key, *members):
        """
        Thêm các thành viên vào Redis Set.
        """
        try:
            self.client.sadd(key, *members)
            print(f"✅ Added {len(members)} members to Set {key}")
        except Exception as err:
            print(f"❌ Error adding members to Set {key}:", err)

    def sismember(self, key, member):
        """
        Kiểm tra một thành viên có thuộc Redis Set không.
        """
        try:
            return self.client.sismember(key, member)
        except Exception as err:
            print(f"❌ Error checking membership for {member} in Set {key}:", err)
            return False

    def flushall(self):
        """
        Xóa tất cả dữ liệu trong Redis.
        """
        try:
            self.client.flushall()
            print("✅ All data flushed from Redis")
        except Exception as err:
            print("❌ Error flushing Redis data:", err)

    def publish(self, channel, message):
        """
        Gửi một thông điệp đến Redis channel.
        """
        try:
            self.client.publish(channel, json.dumps(message))
            print(f"✅ Message published to channel {channel}")
        except Exception as err:
            print(f"❌ Error publishing message to {channel}:", err)

    def subscribe(self, channel, callback):
        """
        Đăng ký lắng nghe thông điệp từ Redis channel.
        """
        try:
            pubsub = self.client.pubsub()
            pubsub.subscribe(**{channel: callback})
            print(f"🔔 Subscribed to channel: {channel}")
            pubsub.run_in_thread(sleep_time=0.1)
        except Exception as err:
            print(f"❌ Error subscribing to channel {channel}:", err)
