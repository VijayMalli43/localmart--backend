# LocalMartAI_Project/core/models.py

from django.db import models
from django.contrib.auth.models import User # Import Django's built-in User model
from decimal import Decimal # Ensure Decimal is imported for calculations

# --- 1. Vendor Model ---
class Vendor(models.Model):
    # REQUIRED: Must be linked to an existing User
    user = models.OneToOneField(User, on_delete=models.CASCADE, related_name='vendor_profile',
                                help_text="Links to the Django User account for vendor login.")
    # REQUIRED
    name = models.CharField(max_length=255, unique=True, help_text="Name of the local grocery shop/vendor.")
    # REQUIRED
    address = models.CharField(max_length=500, help_text="Full street address of the shop.")
    # REQUIRED (has default, but must be provided if no default)
    reliability_score = models.DecimalField(max_digits=3, decimal_places=2, default=0.50,
                                            help_text="AI-derived reliability score (0.00 to 1.00) based on reviews.")
    # OPTIONAL (blank=True, null=True)
    phone = models.CharField(max_length=20, blank=True, null=True, help_text="Vendor's contact phone number.")
    # OPTIONAL (blank=True, null=True)
    email = models.EmailField(max_length=254, blank=True, null=True, help_text="Vendor's contact email address.")
    # REQUIRED (has default, but conceptually required)
    is_active_vendor = models.BooleanField(default=True,
                                           help_text="Indicates if this vendor is currently active and listed on the platform.")
    # Auto-managed by Django
    created_at = models.DateTimeField(auto_now_add=True, help_text="Timestamp when the vendor record was created.")
    # Auto-managed by Django
    updated_at = models.DateTimeField(auto_now=True, help_text="Timestamp when the vendor record was last updated.")

    class Meta:
        verbose_name = "Vendor"
        verbose_name_plural = "Vendors"
        ordering = ['-reliability_score', 'name']

    def __str__(self):
        return self.name

# --- 2. Product Model (Generic Product Information) ---
class Product(models.Model):
    # REQUIRED
    name = models.CharField(max_length=255, unique=True, help_text="e.g., 'Tomato', 'Spinach', 'Milk'.")
    # REQUIRED
    category = models.CharField(max_length=100, help_text="e.g., 'Vegetable', 'Fruit', 'Dairy', 'Staples'.")
    # OPTIONAL (blank=True, null=True)
    description = models.TextField(blank=True, null=True, help_text="Detailed description of the product.")
    # OPTIONAL (blank=True, null=True)
    base_price = models.DecimalField(max_digits=10, decimal_places=2, blank=True, null=True,
                                     help_text="Suggested base price for the product (vendors set their own).")
    # REQUIRED
    unit_of_measure = models.CharField(max_length=50, help_text="e.g., 'kg', 'gram', 'piece', 'bundle', 'litre'.")
    # OPTIONAL (blank=True, null=True)
    image = models.ImageField(upload_to='products/', blank=True, null=True, help_text="Image of the generic product.")
    # Auto-managed by Django
    created_at = models.DateTimeField(auto_now_add=True, help_text="Timestamp when the product record was created.")
    # Auto-managed by Django
    updated_at = models.DateTimeField(auto_now=True, help_text="Timestamp when the product record was last updated.")

    class Meta:
        verbose_name = "Product"
        verbose_name_plural = "Products"
        ordering = ['name']

    def __str__(self):
        return self.name

# --- 3. CompanyWarehouseProducts Model (Central Inventory) ---
class CompanyWarehouseProducts(models.Model):
    # REQUIRED: Must be linked to an existing Product
    product = models.OneToOneField(Product, on_delete=models.CASCADE, related_name='warehouse_stock',
                                   help_text="The generic product available in the company warehouse.")
    # REQUIRED
    current_quantity = models.DecimalField(max_digits=10, decimal_places=2, help_text="Current quantity available in the company warehouse.")
    # REQUIRED
    price = models.DecimalField(max_digits=10, decimal_places=2, help_text="Price of the product from the company warehouse.")
    # REQUIRED (has default, but conceptually required)
    is_available = models.BooleanField(default=True, help_text="Is this product currently in stock at the company warehouse?")
    # Auto-managed by Django
    last_updated = models.DateTimeField(auto_now=True, help_text="Timestamp when this warehouse stock was last updated.")

    class Meta:
        verbose_name = "Company Warehouse Product"
        verbose_name_plural = "Company Warehouse Products"
        ordering = ['product__name']

    def __str__(self):
        return f"{self.product.name} (Warehouse) - Qty: {self.current_quantity} Price: {self.price}"

# --- 4. VendorProduct Model (Vendor-Specific Inventory & Price) ---
class VendorProduct(models.Model):
    # REQUIRED: Must be linked to an existing Vendor
    vendor = models.ForeignKey(Vendor, on_delete=models.CASCADE, related_name='inventory',
                               help_text="The local vendor selling this product.")
    # REQUIRED: Must be linked to an existing Product
    product = models.ForeignKey(Product, on_delete=models.CASCADE, related_name='vendor_offerings',
                                help_text="The generic product being offered by this vendor.")
    # REQUIRED
    current_quantity = models.DecimalField(max_digits=10, decimal_places=2, help_text="Current quantity available at this vendor.")
    # REQUIRED
    price = models.DecimalField(max_digits=10, decimal_places=2, help_text="Price of the product at this vendor.")
    # REQUIRED (has default, but conceptually required)
    is_available = models.BooleanField(default=True, help_text="Is this specific product currently in stock at this vendor?")
    # OPTIONAL (blank=True, null=True)
    ai_freshness_score = models.DecimalField(max_digits=3, decimal_places=2, blank=True, null=True,
                                           help_text="AI-detected freshness score for the current stock batch (0.00 to 1.00).")
    # Auto-managed by Django
    last_updated = models.DateTimeField(auto_now=True, help_text="Timestamp when this vendor's inventory/price was last updated.")

    class Meta:
        unique_together = ('vendor', 'product')
        verbose_name = "Vendor Product (Inventory)"
        verbose_name_plural = "Vendor Products (Inventory)"
        ordering = ['vendor__name', 'product__name']

    def __str__(self):
        return f"{self.product.name} ({self.vendor.name}) - Qty: {self.current_quantity} Price: {self.price}"

# --- 5. Review Model ---
class Review(models.Model):
    # REQUIRED: Must be linked to an existing User (customer)
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='customer_reviews',
                             help_text="The customer (User) who left the review.")
    # REQUIRED: Must be linked to an existing Vendor
    vendor = models.ForeignKey(Vendor, on_delete=models.CASCADE, related_name='reviews',
                               help_text="The vendor being reviewed.")
    # REQUIRED
    rating = models.IntegerField(choices=[(i, str(i)) for i in range(1, 6)],
                                 help_text="Overall rating for the vendor/order from 1 (lowest) to 5 (highest) stars.")
    # OPTIONAL (blank=True)
    comment = models.TextField(blank=True, help_text="Customer's textual review comment. Used by NLP for sentiment/freshness.")
    # OPTIONAL (blank=True, null=True)
    ai_sentiment_score = models.DecimalField(max_digits=3, decimal_places=2, blank=True, null=True,
                                             help_text="AI-derived sentiment score from the comment (e.g., -1.0 for very negative to 1.0 for very positive).")
    # Auto-managed by Django
    created_at = models.DateTimeField(auto_now_add=True, help_text="Timestamp when the review was created.")

    class Meta:
        verbose_name = "Review"
        verbose_name_plural = "Reviews"
        ordering = ['-created_at']

    def __str__(self):
        return f"Review by {self.user.username} for {self.vendor.name} - {self.rating} stars"

# --- 6. Cart Model ---
class Cart(models.Model):
    # REQUIRED: Must be linked to an existing User
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='carts',
                            help_text="The user who owns this cart.")
    # Auto-managed by Django
    created_at = models.DateTimeField(auto_now_add=True)
    # Auto-managed by Django
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        verbose_name = "Cart"
        verbose_name_plural = "Carts"
        ordering = ['-created_at']

    def __str__(self):
        return f"Cart for {self.user.username}"

# --- 7. CartItem Model ---
class CartItem(models.Model):
    # REQUIRED: Must be linked to an existing Cart
    cart = models.ForeignKey(Cart, on_delete=models.CASCADE, related_name='items',
                            help_text="The cart this item belongs to.")
    # REQUIRED: Must be linked to an existing Product
    product = models.ForeignKey(Product, on_delete=models.CASCADE, related_name='cart_items',
                            help_text="The generic product in the cart.")
    # REQUIRED
    quantity = models.DecimalField(max_digits=10, decimal_places=2,
                                help_text="Quantity of the product in the cart (e.g., 0.5 for 500gm, 1.0 for 1kg).")
    # Auto-managed by Django
    # created_at is implicitly part of Cart's update, no direct field here
    # updated_at is implicitly part of Cart's update, no direct field here

    class Meta:
        verbose_name = "Cart Item"
        verbose_name_plural = "Cart Items"
        unique_together = ('cart', 'product')
        ordering = ['product__name']

    def __str__(self):
        return f"{self.quantity} {self.product.unit_of_measure} of {self.product.name} in {self.cart.user.username}'s cart"

# --- 8. Order Model ---
class Order(models.Model):
    # REQUIRED: Must be linked to an existing User
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='orders',
                            help_text="The user who placed this order.")
    # OPTIONAL (null=True, blank=True)
    fulfilled_by_vendor = models.ForeignKey(Vendor, on_delete=models.SET_NULL, null=True, blank=True,
                                            related_name='fulfilled_orders',
                                            help_text="The vendor who fulfilled this order (or null if from company warehouse).")
    # Auto-managed by Django
    order_date = models.DateTimeField(auto_now_add=True)
    # REQUIRED
    total_amount = models.DecimalField(max_digits=10, decimal_places=2, help_text="Total amount of the order.")
    # REQUIRED (has default, but conceptually required)
    status = models.CharField(max_length=50, default='Pending',
                                choices=[
                                    ('Pending', 'Pending'),
                                    ('Processing', 'Processing'),
                                    ('Out for Delivery', 'Out for Delivery'),
                                    ('Delivered', 'Delivered'),
                                    ('Cancelled', 'Cancelled'),
                                ], help_text="Current status of the order.")
    # OPTIONAL (null=True, blank=True)
    fulfillment_details = models.JSONField(null=True, blank=True,
                                            help_text="Stores details of fulfillment (e.g., {'source': 'warehouse', 'items': [...]})")

    class Meta:
        verbose_name = "Order"
        verbose_name_plural = "Orders"
        ordering = ['-order_date']

    def __str__(self):
        return f"Order #{self.id} by {self.user.username} - {self.status}"

# --- 9. OrderItem Model ---
class OrderItem(models.Model):
    # REQUIRED: Must be linked to an existing Order
    order = models.ForeignKey(Order, on_delete=models.CASCADE, related_name='items',
                            help_text="The order this item belongs to.")
    # REQUIRED: Must be linked to an existing Product
    product = models.ForeignKey(Product, on_delete=models.CASCADE, related_name='order_items',
                            help_text="The generic product in the order.")
    # REQUIRED
    quantity = models.DecimalField(max_digits=10, decimal_places=2,
                                help_text="Quantity of the product in the order.")
    # REQUIRED
    price_at_order = models.DecimalField(max_digits=10, decimal_places=2,
                                            help_text="Price of the product at the time of order.")
    # REQUIRED (has choices, but input is required)
    source = models.CharField(max_length=50,
                                choices=[('Warehouse', 'Warehouse'), ('Vendor', 'Vendor')],
                                help_text="Source of this item (Warehouse or specific Vendor).")
    # OPTIONAL (null=True, blank=True)
    source_vendor = models.ForeignKey(Vendor, on_delete=models.SET_NULL, null=True, blank=True,
                                        related_name='sourced_order_items',
                                        help_text="The specific vendor if source is 'Vendor'.")

    class Meta:
        verbose_name = "Order Item"
        verbose_name_plural = "Order Items"
        unique_together = ('order', 'product')
        ordering = ['product__name']

    def __str__(self):
        return f"{self.quantity} {self.product.unit_of_measure} of {self.product.name} in Order #{self.order.id}"

# --- 10. Return Model ---
class Return(models.Model):
    # REQUIRED: Must be linked to an existing OrderItem
    order_item = models.ForeignKey(OrderItem, on_delete=models.CASCADE, related_name='returns',
                                   help_text="The specific order item being returned.")
    # REQUIRED: Must be linked to an existing User
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='returns',
                            help_text="The user initiating the return.")
    # REQUIRED
    return_quantity = models.DecimalField(max_digits=10, decimal_places=2,
                                          help_text="Quantity of the item being returned.")
    # REQUIRED
    return_reason = models.TextField(help_text="Reason for the return.")
    # OPTIONAL (blank=True, null=True)
    return_image = models.ImageField(upload_to='returns/', blank=True, null=True,
                                     help_text="Image of the returned item (e.g., for quality issues).")
    # OPTIONAL (blank=True, null=True)
    ai_return_freshness_prediction = models.CharField(max_length=50, blank=True, null=True,
                                                      help_text="AI's freshness prediction for the returned item (e.g., 'Rotten', 'Healthy').")
    # OPTIONAL (blank=True, null=True)
    ai_prediction_confidence = models.DecimalField(max_digits=5, decimal_places=4, blank=True, null=True,
                                                   help_text="Confidence score of AI's prediction (0.0 to 1.0).")
    # REQUIRED (has default, but conceptually required)
    return_status = models.CharField(max_length=50, default='Pending',
                                   choices=[
                                       ('Pending', 'Pending'),
                                       ('Approved', 'Approved'),
                                       ('Rejected', 'Rejected'),
                                       ('Processed', 'Processed'),
                                   ], help_text="Current status of the return request.")
    # Auto-managed by Django
    requested_at = models.DateTimeField(auto_now_add=True, help_text="Timestamp when the return was requested.")
    # OPTIONAL (null=True, blank=True)
    processed_at = models.DateTimeField(null=True, blank=True, help_text="Timestamp when the return was processed.")

    class Meta:
        verbose_name = "Return"
        verbose_name_plural = "Returns"
        ordering = ['-requested_at']

    def __str__(self):
        return f"Return for OrderItem #{self.order_item.id} by {self.user.username} - {self.return_status}"