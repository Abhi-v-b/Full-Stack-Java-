package com.library.controller;

import com.library.entity.Borrowing;
import com.library.service.BorrowingService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.List;

@RestController
@RequestMapping("/api/borrowings")
@CrossOrigin(origins = "*")
public class BorrowingController {

    private final BorrowingService borrowingService;

    @Autowired
    public BorrowingController(BorrowingService borrowingService) {
        this.borrowingService = borrowingService;
    }

    @PostMapping("/borrow")
    public ResponseEntity<Borrowing> borrowBook(
            @RequestParam Long userId,
            @RequestParam Long bookId,
            @RequestParam(defaultValue = "14") Integer durationInDays) {
        return ResponseEntity.ok(borrowingService.borrowBook(userId, bookId, durationInDays));
    }

    @PutMapping("/{id}/return")
    public ResponseEntity<Borrowing> returnBook(@PathVariable Long id) {
        return ResponseEntity.ok(borrowingService.returnBook(id));
    }

    @GetMapping("/{id}")
    public ResponseEntity<Borrowing> getBorrowingById(@PathVariable Long id) {
        return borrowingService.getBorrowingById(id)
                .map(ResponseEntity::ok)
                .orElse(ResponseEntity.notFound().build());
    }

    @GetMapping
    public ResponseEntity<List<Borrowing>> getAllBorrowings() {
        return ResponseEntity.ok(borrowingService.getAllBorrowings());
    }

    @GetMapping("/user/{userId}")
    public ResponseEntity<List<Borrowing>> getBorrowingsByUser(@PathVariable Long userId) {
        return ResponseEntity.ok(borrowingService.getBorrowingsByUser(userId));
    }

    @GetMapping("/book/{bookId}")
    public ResponseEntity<List<Borrowing>> getBorrowingsByBook(@PathVariable Long bookId) {
        return ResponseEntity.ok(borrowingService.getBorrowingsByBook(bookId));
    }

    @GetMapping("/overdue")
    public ResponseEntity<List<Borrowing>> getOverdueBorrowings() {
        return ResponseEntity.ok(borrowingService.getOverdueBorrowings());
    }

    @GetMapping("/current")
    public ResponseEntity<List<Borrowing>> getCurrentBorrowings() {
        return ResponseEntity.ok(borrowingService.getCurrentBorrowings());
    }

    @GetMapping("/book/{bookId}/available")
    public ResponseEntity<Boolean> isBookAvailableForBorrowing(@PathVariable Long bookId) {
        return ResponseEntity.ok(borrowingService.isBookAvailableForBorrowing(bookId));
    }

    @GetMapping("/user/{userId}/limit")
    public ResponseEntity<Boolean> hasUserExceededBorrowingLimit(@PathVariable Long userId) {
        return ResponseEntity.ok(borrowingService.hasUserExceededBorrowingLimit(userId));
    }

    @PostMapping("/notify-overdue")
    public ResponseEntity<Void> notifyOverdueBooks() {
        borrowingService.notifyOverdueBooks();
        return ResponseEntity.ok().build();
    }
} 