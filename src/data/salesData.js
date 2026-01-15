// src/data/salesData.js
// A sample dataset to populate the AG-Grid table.

export const salesData = [
  { date: '2024-01-15', customer_id: 'CUST001', product_name: 'Widget A', quantity: 5, unit_price: 29.99, region: 'North', sales_rep: 'John Smith' },
  { date: '2024-02-03', customer_id: 'CUST002', product_name: 'Widget B', quantity: 12, unit_price: 45.5, region: 'South', sales_rep: 'Jane Doe' },
  { date: '2024-03-12', customer_id: 'CUST003', product_name: 'Widget C', quantity: 8, unit_price: 19.99, region: 'East', sales_rep: 'Mike Johnson' },
  { date: '2024-03-20', customer_id: 'CUST004', product_name: 'Gadget Pro', quantity: 2, unit_price: 199.99, region: 'West', sales_rep: 'Emily White' },
  { date: '2024-04-05', customer_id: 'CUST005', product_name: 'Widget A', quantity: 10, unit_price: 29.99, region: 'North', sales_rep: 'John Smith' },
  { date: '2024-04-22', customer_id: 'CUST006', product_name: 'Thingamajig', quantity: 15, unit_price: 9.99, region: 'South', sales_rep: 'Jane Doe' },
  { date: '2024-05-10', customer_id: 'CUST007', product_name: 'Widget C', quantity: 20, unit_price: 19.99, region: 'East', sales_rep: 'Mike Johnson' },
  { date: '2024-05-30', customer_id: 'CUST008', product_name: 'Gadget Pro', quantity: 3, unit_price: 199.99, region: 'West', sales_rep: 'Emily White' },
  { date: '2024-06-11', customer_id: 'CUST009', product_name: 'Widget B', quantity: 7, unit_price: 45.5, region: 'North', sales_rep: 'John Smith' },
  { date: '2024-06-28', customer_id: 'CUST010', product_name: 'Thingamajig', quantity: 30, unit_price: 9.99, region: 'South', sales_rep: 'Jane Doe' },
  { date: '2024-07-02', customer_id: 'CUST011', product_name: 'Widget A', quantity: 4, unit_price: 29.99, region: 'East', sales_rep: 'Mike Johnson' },
  { date: '2024-07-19', customer_id: 'CUST012', product_name: 'Gadget Pro', quantity: 1, unit_price: 199.99, region: 'West', sales_rep: 'Emily White' },
];

// Add total_amount to each row
salesData.forEach(row => {
  row.total_amount = row.quantity * row.unit_price;
});